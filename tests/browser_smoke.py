"""Read-only browser verification against an already deployed workstation."""
import argparse
import json
from pathlib import Path
import time
import httpx
from playwright.sync_api import sync_playwright


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8800')
    parser.add_argument('--chromium',type=Path,default=Path.home()/'.cache/vln-chrome/opt/google/chrome/chrome')
    parser.add_argument('--output',type=Path,default=Path('test-results/browser'))
    parser.add_argument('--run',help='Completed generation run to inspect; does not create or modify tasks')
    parser.add_argument('--preview',help='Completed preview associated with --run')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with httpx.Client(base_url=args.url,timeout=15) as client:
        for attempt in range(30):
            try:
                client.get('/api/health').raise_for_status()
                break
            except httpx.HTTPError:
                if attempt==29:raise
                time.sleep(.5)
        evidence={}
        if args.run:
            run=client.get('/api/runs/'+args.run).json()
            assert run['status']=='succeeded',run.get('error')
            evidence['run_id']=args.run
            evidence['quality']=[s['status'] for s in run['generation']['samples']]
            if args.preview:
                preview=client.get('/api/runs/'+args.preview).json()
                artifact=next(a for a in preview['artifacts'] if a['name']=='prepared_samples.json')
                frozen=client.get('/api/artifacts/'+artifact['id']).json()
                assert frozen==run['config']['prepared_samples']
                evidence['preview_exact']=True
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path=str(args.chromium),headless=True,args=['--no-sandbox'])
            page=browser.new_page(viewport={'width':1568,'height':1150})
            errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(args.url+'/',wait_until='networkidle')
            page.wait_for_selector('#sample-list .sample')
            assert page.locator('#atlas').is_visible()
            assert page.locator('.workspace>.panel').count()==3
            evidence['source_records']=int(page.locator('#stat-records').inner_text().replace(',',''))
            assert evidence['source_records']>0
            page.screenshot(path=str(args.output/'atlas.png'),full_page=True)
            page.locator('.platform-nav').get_by_role('link',name='GT 轨迹采集',exact=True).click()
            page.locator('#gt').wait_for(state='visible')
            assert page.locator('#start-gt').is_enabled()
            page.locator('.platform-nav').get_by_role('link',name='数据浏览',exact=True).click()
            page.locator('#atlas').wait_for(state='visible')
            page.set_viewport_size({'width':390,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            page.set_viewport_size({'width':1568,'height':1150})
            if args.run:
                page.goto(args.url+'/#run/'+args.run,wait_until='networkidle')
                page.wait_for_function("document.querySelector('#run-content video')?.readyState>=2")
                assert page.locator('#run-content .status').inner_text()=='已完成'
                page.screenshot(path=str(args.output/'generation.png'),full_page=True)
                page.goto(args.url+'/?origin=seedance',wait_until='networkidle')
                page.wait_for_selector('#detail video')
                assert '非 GT' in page.locator('#detail').inner_text()
                assert not page.locator('#video-coverage').is_visible()
                assert 'LeRobot GT' not in page.locator('footer').inner_text()
                with page.expect_download() as event:
                    page.get_by_role('button',name='导出筛选清单',exact=True).click()
                exported=json.loads(Path(event.value.path()).read_text())
                assert exported['samples']
                assert all(r['origin']=='seedance' and not r['ground_truth'] for r in exported['samples'])
                evidence['exported_generated_samples']=len(exported['samples'])
                page.screenshot(path=str(args.output/'produced.png'),full_page=True)
            assert not errors,errors
            evidence['browser_errors']=errors
            browser.close()
        (args.output/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
        print(json.dumps(evidence,ensure_ascii=False))


if __name__=='__main__':
    main()
