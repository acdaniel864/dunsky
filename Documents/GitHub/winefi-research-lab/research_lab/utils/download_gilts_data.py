import os
import re
import asyncio
from playwright.async_api import async_playwright, expect


async def download_gilts_data():
    """Download UK Gilt data from FRED"""
    # Create directory if it doesn't exist
    target_dir = "data/non_wine_timeseries"
    os.makedirs(target_dir, exist_ok=True)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        await page.goto("https://fred.stlouisfed.org/series/IRLTLT01GBM156N")
        await page.get_by_role("button", name="Download").click()
        
        async with page.expect_download() as download_info:
            await page.get_by_role("link", name="  CSV (data)").click()
        download = await download_info.value
        
        # Save the file to the target directory
        target_path = os.path.join(target_dir, "IRLTLT01GBM156N.csv")
        await download.save_as(target_path)
        print(f"Downloaded file to: {target_path}")

        # ---------------------
        await context.close()
        await browser.close()


def run_download():
    """Synchronous wrapper for the async download function"""
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in a running event loop (like Jupyter), create a task
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(download_gilts_data())
        else:
            # If no event loop is running, we can use asyncio.run
            return asyncio.run(download_gilts_data())
    except ImportError:
        # If nest_asyncio is not available, try a different approach
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(download_gilts_data())
        except RuntimeError:
            # Last resort: create a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(download_gilts_data())
            finally:
                loop.close()


if __name__ == "__main__":
    run_download() 