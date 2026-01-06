from __future__ import annotations

from selenium import webdriver


def driver_config(args) -> webdriver.ChromeOptions:
    """Return ChromeOptions configured from CLI arguments."""
    opts = webdriver.ChromeOptions()
    if getattr(args, "headless", False):
        # Use new headless to ensure screenshots render charts.
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    if getattr(args, "force_device_scale", False):
        opts.add_argument("--force-device-scale-factor=1")
    if getattr(args, "text_only", False):
        opts.add_argument("--blink-settings=imagesEnabled=false")

    download_dir = getattr(args, "download_dir", None)
    if download_dir:
        prefs = {
            "download.default_directory": download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
        }
        opts.add_experimental_option("prefs", prefs)

    return opts
