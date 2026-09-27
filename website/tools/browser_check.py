#!/usr/bin/env python3
"""Browser smoke checks. Requires Playwright + Chromium; optional axe-core script.

python website/tools/browser_check.py --url http://127.0.0.1:4173 \
    --axe /path/to/axe-core/axe.min.js --screenshots /tmp/site-check
"""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:4173')
    parser.add_argument('--axe', type=Path)
    parser.add_argument('--screenshots', type=Path)
    args = parser.parse_args()
    report = {'page_errors': [], 'failed_assets': [], 'viewports': [], 'interactions': []}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={'width':1440,'height':1000}, reduced_motion='reduce')
        page.on('pageerror', lambda e: report['page_errors'].append(str(e)))
        page.on('response', lambda r: report['failed_assets'].append([r.url,r.status]) if r.status>=400 and r.url.startswith(args.url) else None)
        page.goto(args.url, wait_until='networkidle')
        assert page.title().startswith('VLA B-Spline')
        assert page.locator('#hero-video').evaluate('(v)=>v.paused'), 'Reduced-motion autoplay'
        for width in (360,390,768,1024,1440):
            page.set_viewport_size({'width':width,'height':1000})
            assert not page.evaluate('document.documentElement.scrollWidth > innerWidth'), f'Overflow at {width}'
            report['viewports'].append(width)
        page.set_viewport_size({'width':1440,'height':1000})
        original_path = page.locator('#spline-path').get_attribute('d')
        page.locator('#duration-control').fill('0.5')
        assert page.locator('#samples-stat').inner_text()=='12'
        assert page.locator('#duration-stat').inner_text()=='0.60 s'
        page.locator('input[name=rate][value="2"]').check()
        assert page.locator('#samples-stat').inner_text()=='24'
        assert page.locator('#spline-path').get_attribute('d')==original_path
        page.locator('#duration-control').fill('1')
        page.locator('input[name=rate][value="1"]').check()
        report['interactions'].append('duration/rate controls update samples without changing path')
        page.locator('#tab-clock').click()
        assert page.locator('#result-value').inner_text()=='+0.379'
        page.locator('#tab-clock').press('ArrowRight')
        assert page.locator('#tab-queries').get_attribute('aria-selected')=='true'
        assert page.locator('#result-value').inner_text()=='2.24–2.68×'
        page.locator('#tab-language').click()
        assert page.locator('#result-value').inner_text()=='+9.33 pp'
        with page.expect_download() as download:
            page.locator('#download-results').click()
        assert download.value.suggested_filename=='vla-bspline-selected-results.csv'
        csv=Path(download.value.path()).read_text()
        assert len(csv.splitlines())==14
        report['interactions'].append('result tabs, keyboard navigation and 13-row CSV export')
        page.locator('#comparison-play').click()
        page.wait_for_function('document.querySelector("#baseline-video").currentTime > 1.5')
        before, after = page.evaluate('[document.querySelector("#baseline-video").currentTime,document.querySelector("#phase-video").currentTime]')
        assert abs(before-after)<0.5, (before,after)
        page.locator('#comparison-play').click()
        assert page.locator('#baseline-video').evaluate('(v)=>v.paused')
        page.locator('#comparison-scrub').fill('90')
        page.wait_for_timeout(300)
        assert page.locator('#phase-video').evaluate('(v)=>v.currentTime>v.duration-0.2'), 'Completed video must hold final frame'
        page.locator('#demo-tab-rinse').click()
        assert '444' in page.locator('#phase-caption').inner_text()
        assert 'scene 1015' in page.locator('#demo-protocol').inner_text()
        page.locator('#comparison-play').click()
        page.wait_for_function('document.querySelector("#baseline-video").currentTime > 0.5')
        page.locator('#comparison-play').click()
        page.locator('#demo-tab-kettle').click()
        report['interactions'].append('paired playback, seek, final-frame hold, task switch')
        if args.axe:
            page.add_script_tag(path=str(args.axe))
            violations=page.evaluate('async()=> (await axe.run(document,{runOnly:{type:"tag",values:["wcag2a","wcag2aa","wcag21aa"]}})).violations')
            report['accessibility']=[{'id':v['id'],'impact':v['impact'],'nodes':[{'target':n['target'],'summary':n.get('failureSummary')} for n in v['nodes']]} for v in violations]
        if args.screenshots:
            args.screenshots.mkdir(parents=True,exist_ok=True)
            page.evaluate('window.scrollTo(0,0)')
            page.screenshot(path=str(args.screenshots/'desktop.png'),full_page=True)
            page.screenshot(path=str(args.screenshots/'desktop-hero.png'))
            page.set_viewport_size({'width':390,'height':844})
            page.screenshot(path=str(args.screenshots/'mobile.png'),full_page=True)
            page.screenshot(path=str(args.screenshots/'mobile-hero.png'))
        browser.close()
    print(json.dumps(report,indent=2))
    assert not report['page_errors'] and not report['failed_assets']
    assert not report.get('accessibility'), 'Accessibility checks found violations'


if __name__=='__main__':
    main()
