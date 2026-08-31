"""
iran_exchange_adapter.py — SKELETON ONLY. Places orders via browser automation
(Playwright) because the exchange has no API. You MUST fill in the selectors
marked TODO by inspecting the real site with your browser's DevTools.

Install:  pip install playwright  &&  playwright install chromium

Security:
  - NEVER hardcode username/password/2FA secrets in this file.
  - Read them from environment variables (set outside of any git repo).
  - Log in once per session and reuse the browser context; re-logging in
    repeatedly is what usually triggers CAPTCHAs / bot-detection / bans.
"""
import os
import time
from playwright.sync_api import sync_playwright

EXCHANGE_URL = os.environ.get("IRAN_EXCHANGE_URL", "https://REPLACE_WITH_YOUR_EXCHANGE_URL")
USERNAME = os.environ.get("IRAN_EXCHANGE_USER")
PASSWORD = os.environ.get("IRAN_EXCHANGE_PASS")


class IranExchangeAdapter:
    def __init__(self, headless=True, dry_run=True):
        self.dry_run = dry_run  # <-- default TRUE. Flip only when you're sure.
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=headless)
        self.page = self.browser.new_page()
        self.logged_in = False

    def login(self):
        if not USERNAME or not PASSWORD:
            raise RuntimeError("Set IRAN_EXCHANGE_USER / IRAN_EXCHANGE_PASS env vars first.")
        self.page.goto(EXCHANGE_URL)
        # TODO: replace these selectors with the real ones from your exchange's login page
        self.page.fill('input[name="username"]', USERNAME)          # TODO
        self.page.fill('input[name="password"]', PASSWORD)          # TODO
        self.page.click('button[type="submit"]')                    # TODO
        # TODO: if the site uses 2FA/SMS/CAPTCHA, you likely cannot fully automate
        # this step safely/reliably — consider a manual login + persistent
        # browser profile (see note at bottom of file) instead of automating it.
        self.page.wait_for_timeout(3000)
        self.logged_in = True

    def get_balance(self, asset="USDT") -> float:
        # TODO: navigate to the wallet/balance page and parse the number shown
        raise NotImplementedError("Fill in the real selector/parsing for your exchange.")

    def place_market_order(self, side: str, symbol: str, amount: float):
        """side: 'buy' or 'sell'. amount: quantity in base asset (or IRT value — depends
        on how the exchange's order form is structured; check carefully)."""
        assert side in ("buy", "sell")
        if self.dry_run:
            print(f"[DRY RUN] would place {side.upper()} {amount} {symbol}")
            return {"status": "dry_run", "side": side, "amount": amount, "symbol": symbol}

        # TODO: real flow, e.g.:
        # self.page.click(f'text="{symbol}"')
        # self.page.click(f'text="{"خرید" if side=="buy" else "فروش"}"')
        # self.page.fill('input[name="amount"]', str(amount))
        # self.page.click('button:has-text("ثبت سفارش")')
        # self.page.wait_for_selector('text="سفارش با موفقیت ثبت شد"', timeout=15000)
        raise NotImplementedError("Fill in the real click/selector flow for your exchange.")

    def close(self):
        self.browser.close()
        self._pw.stop()


# NOTE on 2FA/CAPTCHA-heavy exchanges:
# Many Iranian exchanges add CAPTCHA or SMS-OTP specifically to block bots like
# this one. If that's the case here, a more realistic (and much more robust)
# pattern is: log in manually ONE time in a persistent Playwright browser
# profile (`launch_persistent_context`), leave that session open/reused by the
# bot, and only automate the order-placement clicks, not the login itself.
