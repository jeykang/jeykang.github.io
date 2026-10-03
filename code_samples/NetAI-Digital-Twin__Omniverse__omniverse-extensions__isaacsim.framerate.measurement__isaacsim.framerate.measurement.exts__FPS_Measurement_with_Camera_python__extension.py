# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import omni.ext
import omni.ui as ui
import omni.kit.menu.utils
from omni.kit.menu.utils import MenuItemDescription

from .global_variables import EXTENSION_TITLE
from .ui_builder import UIBuilder

class Extension(omni.ext.IExt):
    """
    Main Entry Point for the Extension.
    Inherits from omni.ext.IExt to hook into Omniverse Kit's startup/shutdown lifecycle.
    """
    
    def on_startup(self, ext_id: str):
        """
        Called when the extension is loaded.
        Initializes the extension state and builds the menu.
        """
        print(f"[{EXTENSION_TITLE}] Startup")
        
        self._ext_id = ext_id
        self._window = None
        self._ui_builder = UIBuilder()
        self._menu_items = []
        
        # Create and register "KKR-Tools" tab on the menu bar
        self._build_menu()

    def on_shutdown(self):
        """
        Called when the extension is disabled or Isaac Sim closes.
        Cleans up resources to prevent memory leaks or UI artifacts.
        """
        print(f"[{EXTENSION_TITLE}] Shutdown")
        
        # 1. Remove menu items
        if self._menu_items:
            omni.kit.menu.utils.remove_menu_items(self._menu_items, "KKR-Tools")
        
        # 2. Destroy the window
        if self._window:
            self._window.destroy()
            self._window = None
            
        # 3. Cleanup UIBuilder
        if self._ui_builder:
            self._ui_builder.cleanup()
            self._ui_builder = None

    def _build_menu(self):
        """
        Registers the current extension to the KKR-Tools menu tab.
        """
        # Create MenuItemDescription
        menu_items = [
            MenuItemDescription(
                name=EXTENSION_TITLE,
                onclick_fn=self._toggle_window
            )
        ]
        
        # Create a top-level menu named "KKR-Tools" and add items to it.
        # Extensions sharing this name will be grouped together.
        omni.kit.menu.utils.add_menu_items(menu_items, "KKR-Tools")
        self._menu_items = menu_items

    def _build_window(self):
        """Create window (Floating mode)"""
        # dockPreference=ui.DockPreference.DISABLED -> Create a floating window that is not docked
        self._window = ui.Window(
            title=EXTENSION_TITLE,
            width=400,
            height=550,
            dockPreference=ui.DockPreference.DISABLED
        )
        
        # Handle visibility when the window is closed (X button)
        self._window.set_visibility_changed_fn(self._on_window_visibility_changed)

        # Connect the content of UIBuilder to the window frame
        with self._window.frame:
            # Configure to allow scrolling if content exceeds window size
            with ui.ScrollingFrame(
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED
            ):
                self._ui_builder.build_ui()

    def _toggle_window(self, *args):
        """Executed when menu is clicked: Toggle window open/close"""
        if not self._window:
            self._build_window()
            self._window.visible = True
        else:
            self._window.visible = not self._window.visible

    def _on_window_visibility_changed(self, visible):
        """Called when the window's X button is pressed or visibility changes"""
        # Add logic here if you need to pause/resume recording when window hides.
        pass