"""Headless interaction checks. Exported fixture is automated test data, never human labels."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from playwright.sync_api import sync_playwright


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-clips', type=int, default=26)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh output required')
    with sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True, args=['--no-sandbox', '--autoplay-policy=no-user-gesture-required'])
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(a.url)
        page.wait_for_function('decoded !== null')
        assert page.locator('#completed').inner_text() == f'已有效记录 0 / {a.expected_clips}'
        assert page.locator('#width').inner_text() == ''
        assert page.evaluate('DATA.records.length') == a.expected_clips
        assert page.evaluate('DATA.records.every(r => !Object.keys(r).some(k => /score|event|boundary|keyword_end/.test(k)))')
        page.locator('#audio').evaluate('(audio) => audio.play()')
        page.wait_for_function('document.getElementById("audio").currentTime > .1')
        page.locator('#audio').evaluate('(audio) => audio.pause()')
        page.select_option('#speed', '0.75')
        assert page.locator('#audio').evaluate('(audio) => audio.playbackRate') == .75
        page.locator('#reviewer').fill('AUTOMATED_UI_TEST_NOT_HUMAN_LABELS')
        page.locator('#lower').fill('0.5')
        page.locator('#upper').fill('0.49')
        page.select_option('#status', 'matches')
        assert page.locator('#completed').inner_text() == f'已有效记录 0 / {a.expected_clips}'
        assert '上限不能早于下限' in page.locator('#message').inner_text()
        page.locator('#upper').fill('0.51')
        assert page.locator('#completed').inner_text() == f'已有效记录 1 / {a.expected_clips}'
        page.click('#zoom')
        assert page.evaluate('Math.abs(view[0]-.3)<1e-9 && Math.abs(view[1]-.71)<1e-9')
        page.click('#full')
        page.click('#next')
        page.wait_for_function('decoded !== null')
        page.click('#previous')
        page.wait_for_function('decoded !== null')
        assert float(page.locator('#lower').input_value()) == .5
        page.reload()
        page.wait_for_function('decoded !== null')
        assert page.locator('#completed').inner_text() == f'已有效记录 1 / {a.expected_clips}'
        assert page.locator('#reviewer').input_value() == 'AUTOMATED_UI_TEST_NOT_HUMAN_LABELS'
        with page.expect_download() as info:
            page.click('#export')
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'automated-fixture.json'
            info.value.save_as(target)
            exported = json.loads(target.read_text())
        assert exported['review_complete'] is False
        assert exported['detector_outputs_shown'] is False
        assert exported['automatic_boundaries_shown'] is False
        assert len(exported['records']) == a.expected_clips
        assert exported['records'][0]['lower_sample'] == 8000
        assert exported['records'][0]['upper_sample'] == 8160
        assert not errors, errors
        version = browser.version
        context.close()  # Discards test-only localStorage and all synthetic annotations.
        browser.close()
    result = dict(role='Automated UI checks only; no human endpoint labels generated',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  browser=version, clips=a.expected_clips, audio_playback_and_decode_passed=True,
                  invalid_interval_rejected=True, sample_exact_export_passed=True,
                  navigation_persistence_and_reload_passed=True, partial_export_not_marked_complete=True,
                  no_detector_or_automatic_boundaries=True, browser_errors=errors,
                  test_context_discarded=True, human_annotations_present=False)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
