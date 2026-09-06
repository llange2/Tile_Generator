"""Tkinter GUI for the tile generator.

Pick a base terrain tile (e.g. grass) and an overlay terrain tile (e.g. water),
choose an output tile size, and generate the full set of corner-blended
transition tiles between them.
"""

from __future__ import annotations

import json
import os
import random
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk

import tile_blend as tb

THUMB = 96
SOURCE_THUMB = 84
EXAMPLE_TILE_PX = 64
ISO_EXAMPLE_TILE_W = EXAMPLE_TILE_PX * 2
ISO_EXAMPLE_TILE_H = EXAMPLE_TILE_PX

SETTINGS_PATH = Path(__file__).resolve().parent / "tile_generator_settings.json"
ATLAS_FILENAME = "tileset.json"

IMAGE_FILETYPES = [
    ("Image files", "*.bmp *.png *.gif *.jpg *.jpeg *.tga"),
    ("All files", "*.*"),
]


def _center_window(win, width=None, height=None):
    """Position win so it opens at the center of the desktop.

    If width/height are omitted, the window's own requested size is used
    (for windows like Toplevels whose size comes from their content).
    """
    win.update_idletasks()
    w = width if width is not None else win.winfo_reqwidth()
    h = height if height is not None else win.winfo_reqheight()
    screen_w = win.winfo_screenwidth()
    screen_h = win.winfo_screenheight()
    x = (screen_w - w) // 2
    y = (screen_h - h) // 2
    if width is not None and height is not None:
        win.geometry(f"{w}x{h}+{x}+{y}")
    else:
        win.geometry(f"+{x}+{y}")


class SourcePicker(ttk.LabelFrame):
    def __init__(self, master, title, on_change, clearable=False):
        super().__init__(master, text=title, padding=8)
        self.on_change = on_change
        self.path = None
        self.image = None  # PIL RGBA

        self.canvas = tk.Canvas(self, width=SOURCE_THUMB, height=SOURCE_THUMB,
                                 background="#222", highlightthickness=1,
                                 highlightbackground="#555")
        self.canvas.grid(row=0, column=0, rowspan=2, padx=(0, 8))
        self._photo = None

        self.path_var = tk.StringVar(value="(no file selected)")
        ttk.Label(self, textvariable=self.path_var, width=28, wraplength=180).grid(
            row=0, column=1, sticky="w")
        btn_row = ttk.Frame(self)
        btn_row.grid(row=1, column=1, sticky="w", pady=(4, 0))
        ttk.Button(btn_row, text="Browse...", command=self.browse).pack(side="left")
        if clearable:
            ttk.Button(btn_row, text="Clear", command=self.clear).pack(side="left", padx=(4, 0))

    def browse(self):
        path = filedialog.askopenfilename(title="Select tile image", filetypes=IMAGE_FILETYPES)
        if not path:
            return
        self.set_from_path(path)

    def set_from_path(self, path: str) -> bool:
        try:
            img = tb.load_image(path)
        except Exception as exc:
            messagebox.showerror("Could not load image", f"{path}\n\n{exc}")
            return False
        self.path = path
        self.image = img
        self.path_var.set(f"{os.path.basename(path)}  ({img.width}x{img.height})")
        self._draw_thumb(img)
        self.on_change()
        return True

    def clear(self):
        self.path = None
        self.image = None
        self.path_var.set("(no file selected)")
        self.canvas.delete("all")
        self.on_change()

    def _draw_thumb(self, img: Image.Image):
        thumb = img.copy()
        thumb.thumbnail((SOURCE_THUMB, SOURCE_THUMB), Image.LANCZOS)
        canvas_img = Image.new("RGBA", (SOURCE_THUMB, SOURCE_THUMB), (34, 34, 34, 255))
        ox = (SOURCE_THUMB - thumb.width) // 2
        oy = (SOURCE_THUMB - thumb.height) // 2
        canvas_img.paste(thumb, (ox, oy), thumb)
        self._photo = ImageTk.PhotoImage(canvas_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

    @property
    def stem(self):
        if not self.path:
            return "tile"
        return os.path.splitext(os.path.basename(self.path))[0]


class TileGeneratorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tile Generator")
        self.minsize(860, 560)

        self.results = None  # dict[code -> PIL.Image] from last generation
        self._thumb_refs = []
        self.last_output_folder = None

        self._build_layout()
        self._load_settings()

        self.update_idletasks()
        _center_window(self, 980, self.winfo_reqheight() + 10)

    # ------------------------------------------------------------------ UI

    def _build_layout(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        # --- Source pickers (top) ---
        sources = ttk.Frame(root)
        sources.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        sources.columnconfigure(0, weight=1)
        sources.columnconfigure(1, weight=1)
        sources.columnconfigure(2, weight=1)

        self.picker_a = SourcePicker(sources, "Terrain A (base)", self._on_source_change)
        self.picker_a.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.picker_b = SourcePicker(sources, "Terrain B (overlay)", self._on_source_change)
        self.picker_b.grid(row=0, column=1, sticky="ew", padx=4)
        self.picker_c = SourcePicker(sources, "Transition (optional, e.g. sand)",
                                      self._on_source_change, clearable=True)
        self.picker_c.grid(row=0, column=2, sticky="ew", padx=(4, 0))

        # --- Options (left) ---
        opts = ttk.LabelFrame(root, text="Options", padding=10)
        opts.grid(row=1, column=0, sticky="nsew", padx=(0, 10))

        row = 0
        ttk.Label(opts, text="Output tile size (px)").grid(row=row, column=0, sticky="w")
        self.size_var = tk.IntVar(value=48)
        ttk.Spinbox(opts, from_=8, to=1024, increment=8, textvariable=self.size_var,
                    width=8).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(opts, text="Resample method").grid(row=row, column=0, sticky="w", pady=(6, 0))
        self.resample_var = tk.StringVar(value="Lanczos")
        ttk.Combobox(opts, textvariable=self.resample_var, state="readonly", width=10,
                     values=list(tb.RESAMPLE_METHODS.keys())).grid(
            row=row, column=1, sticky="w", pady=(6, 0))
        row += 1

        row = self._add_slider(opts, row, "Roughness (coastline noise)", "roughness_var", 0.35, 0.0, 1.0)
        row = self._add_slider(opts, row, "Edge softness (% of tile size)", "feather_var", 4.0, 0.0, 50.0)
        row = self._add_slider(opts, row, "Island / pond size", "blob_var", 0.28, *tb.BLOB_SIZE_RANGE)
        row = self._add_slider(opts, row, "Transition width (% of tile size)", "transition_var", 3.0, 0.0, 25.0)

        ttk.Label(opts, text="Seed").grid(row=row, column=0, sticky="w", pady=(6, 0))
        seed_frame = ttk.Frame(opts)
        seed_frame.grid(row=row, column=1, sticky="w", pady=(6, 0))
        self.seed_var = tk.IntVar(value=1)
        ttk.Entry(seed_frame, textvariable=self.seed_var, width=8).pack(side="left")
        ttk.Button(seed_frame, text="Randomize", command=self._randomize_seed).pack(
            side="left", padx=(4, 0))
        row += 1

        self.basic_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Include basic tiles",
                         variable=self.basic_var).grid(row=row, column=0, columnspan=2,
                                                        sticky="w", pady=(8, 0))
        row += 1

        self.diag_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Include diagonal (checkerboard) combinations",
                         variable=self.diag_var).grid(row=row, column=0, columnspan=2,
                                                       sticky="w")
        row += 1

        self.specials_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Include island / pond tiles",
                         variable=self.specials_var).grid(row=row, column=0, columnspan=2,
                                                           sticky="w")
        row += 1

        self.border_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Include seamless border tile",
                         variable=self.border_var).grid(row=row, column=0, columnspan=2,
                                                         sticky="w")
        row += 1

        self.include_pure_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Include resized copies of the source tiles",
                         variable=self.include_pure_var).grid(row=row, column=0, columnspan=2,
                                                                sticky="w")
        row += 1

        ttk.Separator(opts, orient="horizontal").grid(row=row, column=0, columnspan=2,
                                                        sticky="ew", pady=8)
        row += 1

        self.sequential_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Sequential numbering",
                         variable=self.sequential_var, command=self._toggle_sequential).grid(
            row=row, column=0, columnspan=2, sticky="w")
        row += 1

        ttk.Label(opts, text="Start index").grid(row=row, column=0, sticky="w")
        self.start_index_var = tk.IntVar(value=37)
        self.start_index_entry = ttk.Spinbox(opts, from_=0, to=9999, textvariable=self.start_index_var,
                                              width=8, state="disabled")
        self.start_index_entry.grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(opts, text="Output format").grid(row=row, column=0, sticky="w", pady=(6, 0))
        self.format_var = tk.StringVar(value="Same as source")
        ttk.Combobox(opts, textvariable=self.format_var, state="readonly", width=14,
                     values=["Same as source", "PNG", "BMP"]).grid(
            row=row, column=1, sticky="w", pady=(6, 0))
        row += 1

        ttk.Separator(opts, orient="horizontal").grid(row=row, column=0, columnspan=2,
                                                        sticky="ew", pady=8)
        row += 1

        shape_frame = ttk.LabelFrame(opts, text="Tile shape", padding=8)
        shape_frame.grid(row=row, column=0, columnspan=2, sticky="ew")
        self.isometric_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(shape_frame, text="Isometric tiles", variable=self.isometric_var,
                         style="Toolbutton").pack(anchor="w")
        row += 1

        ttk.Separator(opts, orient="horizontal").grid(row=row, column=0, columnspan=2,
                                                        sticky="ew", pady=8)
        row += 1

        self.generate_btn = ttk.Button(opts, text="Generate Preview", command=self.generate_preview)
        self.generate_btn.grid(row=row, column=0, columnspan=2, sticky="ew")
        row += 1

        self.example_btn = ttk.Button(opts, text="View Example Tiling", command=self.view_example_tiling,
                                       state="disabled")
        self.example_btn.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        row += 1

        self.export_btn = ttk.Button(opts, text="Export Tiles...", command=self.export_tiles,
                                      state="disabled")
        self.export_btn.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        row += 1

        self.status_var = tk.StringVar(value="Select two source tiles to begin.")
        ttk.Label(opts, textvariable=self.status_var, wraplength=260,
                  foreground="#555").grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 0))

        # --- Preview grid (right) ---
        preview_frame = ttk.LabelFrame(root, text="Generated tiles", padding=6)
        preview_frame.grid(row=1, column=1, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(preview_frame, background="#f2f2f2", highlightthickness=0)
        vsb = ttk.Scrollbar(preview_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        self.grid_holder = ttk.Frame(self.canvas)
        self._grid_window = self.canvas.create_window((0, 0), window=self.grid_holder, anchor="nw")
        self.grid_holder.bind("<Configure>",
                               lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>",
                          lambda e: self.canvas.itemconfigure(self._grid_window, width=e.width))
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _add_slider(self, parent, row, label, attr_name, default, lo, hi):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(6, 0))
        var = tk.DoubleVar(value=default)
        setattr(self, attr_name, var)
        scale = ttk.Scale(parent, from_=lo, to=hi, variable=var, orient="horizontal", length=110)
        scale.grid(row=row, column=1, sticky="w", pady=(6, 0))
        base = attr_name[:-4] if attr_name.endswith("_var") else attr_name
        setattr(self, base + "_scale", scale)
        return row + 1

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def _toggle_sequential(self):
        self.start_index_entry.configure(state="normal" if self.sequential_var.get() else "disabled")

    def _randomize_seed(self):
        self.seed_var.set(random.randint(0, 999999))

    def _on_source_change(self):
        if self.picker_a.image is not None and self.size_var.get() == 48:
            self.size_var.set(self.picker_a.image.width)
        self.status_var.set("Ready to generate.")

    # ------------------------------------------------------------- actions

    def _collect_options(self) -> tb.BlendOptions:
        size = int(self.size_var.get())
        return tb.BlendOptions(
            size=size,
            resample=self.resample_var.get(),
            noise_strength=float(self.roughness_var.get()),
            feather_px=float(self.feather_var.get()) / 100.0 * size,
            edge_margin_frac=0.18,
            include_basic=bool(self.basic_var.get()),
            include_diagonals=bool(self.diag_var.get()),
            include_specials=bool(self.specials_var.get()),
            include_border=bool(self.border_var.get()),
            blob_radius_frac=float(self.blob_var.get()),
            blob_feather_frac=0.10,
            seed=int(self.seed_var.get()),
            transition_width_px=float(self.transition_var.get()) / 100.0 * size,
        )

    def _current_filenames(self, ext: str) -> dict:
        codes = [c for c, _b, _cat in tb.STANDARD_COMBOS] if self.basic_var.get() else []
        if self.diag_var.get():
            codes += [c for c, _b, _cat in tb.DIAGONAL_COMBOS]
        if self.specials_var.get():
            codes += ["island", "pond"]
        if self.border_var.get():
            codes += ["border"]
        use_transition = self.picker_c.image is not None and float(self.transition_var.get()) > 0
        if self.include_pure_var.get():
            codes += ["_pure_a", "_pure_b"]
            if use_transition:
                codes += ["_pure_c"]

        if self.sequential_var.get():
            names = tb.sequential_filenames(int(self.start_index_var.get()), self.picker_b.stem, ext)
            if self.diag_var.get():
                next_idx = int(self.start_index_var.get()) + len(tb.LEGACY_ORDER)
                overlay_stem = tb.strip_leading_index(self.picker_b.stem)
                for code in ("diag-tlbr", "diag-trbl"):
                    names[code] = f"{next_idx:03d}_{overlay_stem}{ext}"
                    next_idx += 1
            if not self.basic_var.get():
                for code, _b, _cat in tb.STANDARD_COMBOS:
                    names.pop(code, None)
            if not self.specials_var.get():
                names.pop("island", None)
                names.pop("pond", None)
            if self.border_var.get():
                names.setdefault("border", f"{self.picker_a.stem}_full_1{ext}")
            if not self.include_pure_var.get():
                names.pop("_pure_a", None)
                names.pop("_pure_b", None)
            elif "_pure_a" not in names:
                names["_pure_a"] = f"{self.picker_a.stem}_full{ext}"
            if use_transition and self.include_pure_var.get():
                names["_pure_c"] = f"{self.picker_c.stem}_full{ext}"
            return {c: names[c] for c in codes if c in names}

        c_stem = self.picker_c.stem if self.picker_c.path else "transition"
        names = tb.descriptive_filenames(self.picker_a.stem, self.picker_b.stem, codes, ext, c_stem=c_stem)
        if self.isometric_var.get():
            # Overrides whichever codes the isometric edge-naming convention
            # can represent (edge-*/inner-*/pure tiles); corner-*, diag-*,
            # island and pond have no clean diamond-edge meaning (see
            # tile_blend.EDGE_ATTR_MAP) and keep their descriptive names.
            names.update(tb.isometric_edge_filenames(self.picker_a.stem, self.picker_b.stem, codes, ext))
        return names

    def generate_preview(self):
        if self.picker_a.image is None or self.picker_b.image is None:
            messagebox.showwarning("Missing sources", "Please select both Terrain A and Terrain B images first.")
            return
        try:
            opts = self._collect_options()
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Invalid options", str(exc))
            return

        self.status_var.set("Generating...")
        self.update_idletasks()
        try:
            self.results = tb.generate_full_set(self.picker_a.image, self.picker_b.image, opts,
                                                 img_c=self.picker_c.image)
            if self.isometric_var.get():
                self.results = {code: tb.to_isometric(img) for code, img in self.results.items()}
        except Exception as exc:
            messagebox.showerror("Generation failed", str(exc))
            self.status_var.set("Generation failed.")
            return

        self._render_preview_grid()
        self.export_btn.configure(state="normal")
        self.example_btn.configure(state="normal")
        self.status_var.set(f"Generated {len(self.results)} tiles at {opts.size}x{opts.size}px.")
        self._save_settings()

    def _render_preview_grid(self):
        for child in self.grid_holder.winfo_children():
            child.destroy()
        self._thumb_refs.clear()

        names = self._current_filenames(".png")
        order = [c for c, _b, _cat in tb.STANDARD_COMBOS] if self.basic_var.get() else []
        if self.diag_var.get():
            order += [c for c, _b, _cat in tb.DIAGONAL_COMBOS]
        if self.specials_var.get():
            order += ["island", "pond"]
        if self.border_var.get():
            order += ["border"]
        if self.include_pure_var.get():
            order += ["_pure_a", "_pure_b", "_pure_c"]

        cols = 4
        for i, code in enumerate(order):
            if code not in self.results:
                continue
            img = self.results[code]
            thumb = img.copy()
            thumb.thumbnail((THUMB, THUMB), Image.NEAREST)
            photo = ImageTk.PhotoImage(thumb)
            self._thumb_refs.append(photo)

            cell = ttk.Frame(self.grid_holder, padding=4)
            cell.grid(row=i // cols, column=i % cols, sticky="n")
            tk.Label(cell, image=photo, background="#ddd", borderwidth=1,
                     relief="solid").pack()
            label_text = names.get(code, code)
            ttk.Label(cell, text=label_text, font=("Segoe UI", 8)).pack()
            ttk.Label(cell, text=tb.CODE_LABELS.get(code, code), font=("Segoe UI", 7),
                      foreground="#666", wraplength=THUMB).pack()

    def _build_grid_sheet(self, codes) -> Image.Image:
        rows, cols = len(codes), len(codes[0])
        px = EXAMPLE_TILE_PX
        sheet = Image.new("RGBA", (cols * px, rows * px))
        for r, row_codes in enumerate(codes):
            for c, code in enumerate(row_codes):
                img = self.results.get(code)
                if img is None:
                    continue
                sheet.paste(img.resize((px, px), Image.NEAREST), (c * px, r * px))
        return sheet

    def _build_isometric_sheet(self, codes) -> Image.Image:
        """Lays diamond tiles out in the classic staggered isometric grid.

        Cell (r, c) is placed at screen position ((c - r) * tile_w/2,
        (c + r) * tile_h/2) so neighboring diamonds interlock edge-to-edge
        instead of sitting in a plain square grid.
        """
        rows, cols = len(codes), len(codes[0])
        tw, th = ISO_EXAMPLE_TILE_W, ISO_EXAMPLE_TILE_H
        half_w, half_h = tw / 2.0, th / 2.0

        positions = {}
        xs, ys = [], []
        for r in range(rows):
            for c in range(cols):
                x = (c - r) * half_w
                y = (c + r) * half_h
                positions[(r, c)] = (x, y)
                xs.append(x)
                ys.append(y)

        offset_x, offset_y = -min(xs), -min(ys)
        canvas_w = int(round(max(xs) + offset_x + tw))
        canvas_h = int(round(max(ys) + offset_y + th))
        sheet = Image.new("RGBA", (canvas_w, canvas_h))

        for r, row_codes in enumerate(codes):
            for c, code in enumerate(row_codes):
                img = self.results.get(code)
                if img is None:
                    continue
                tile = img.resize((tw, th), Image.NEAREST)
                x, y = positions[(r, c)]
                sheet.paste(tile, (int(round(x + offset_x)), int(round(y + offset_y))), tile)
        return sheet

    def view_example_tiling(self):
        if not self.results:
            return

        include_diagonals = "diag-tlbr" in self.results and "diag-trbl" in self.results
        codes = tb.example_layout_codes(include_diagonals=include_diagonals)

        # Codes outside the corner-combo system (island/pond/transition
        # swatch) get dropped onto a cell that would otherwise just be a
        # plain pure_b/pure_a fill, so they still read as part of the scene.
        for r, c, code in ((5, 5, "island"), (1, 1, "pond"), (2, 8, "_pure_c")):
            if code in self.results:
                codes[r][c] = code

        if self.isometric_var.get():
            sheet = self._build_isometric_sheet(codes)
        else:
            sheet = self._build_grid_sheet(codes)

        if getattr(self, "_example_win", None) is not None and self._example_win.winfo_exists():
            self._example_win.destroy()

        win = tk.Toplevel(self)
        win.title("Example Tiling")
        win.resizable(False, False)
        self._example_win = win

        ttk.Label(win, text="Static example layout using every generated tile at least once.",
                  padding=8).pack()
        canvas = tk.Canvas(win, width=sheet.width, height=sheet.height, highlightthickness=0)
        canvas.pack(padx=8, pady=(0, 8))
        photo = ImageTk.PhotoImage(sheet)
        canvas.image = photo  # keep a reference alive
        canvas.create_image(0, 0, anchor="nw", image=photo)

        _center_window(win)

    def export_tiles(self):
        if not self.results:
            return
        folder = filedialog.askdirectory(title="Choose output folder",
                                          initialdir=self.last_output_folder or os.getcwd())
        if not folder:
            return
        self.last_output_folder = folder

        fmt = self.format_var.get()
        b_ext = os.path.splitext(self.picker_b.path or "")[1] or ".png"
        ext = {"Same as source": b_ext, "PNG": ".png", "BMP": ".bmp"}.get(fmt, b_ext)

        names = self._current_filenames(ext)
        saved = 0
        errors = []
        for code, filename in names.items():
            img = self.results.get(code)
            if img is None:
                continue
            out_path = os.path.join(folder, filename)
            try:
                to_save = img.convert("RGB") if ext.lower() in (".bmp", ".jpg", ".jpeg") else img
                to_save.save(out_path)
                saved += 1
            except Exception as exc:
                errors.append(f"{filename}: {exc}")

        atlas_note = ""
        if self.isometric_var.get():
            added = self._update_atlas(folder, names, errors)
            if added:
                atlas_note = f"\nAdded {added} new entr{'y' if added == 1 else 'ies'} to {ATLAS_FILENAME}."

        if errors:
            messagebox.showwarning("Export finished with errors",
                                    f"Saved {saved} tiles to:\n{folder}\n\nErrors:\n" + "\n".join(errors))
        else:
            messagebox.showinfo("Export complete", f"Saved {saved} tiles to:\n{folder}{atlas_note}")
        self.status_var.set(f"Exported {saved} tiles to {folder}")
        self._save_settings()

    def _update_atlas(self, folder: str, names: dict, errors: list) -> int:
        """Merges this export's edge-representable tiles into <folder>/tileset.json,
        skipping filenames already present so re-exporting never duplicates entries.
        Returns the number of newly added entries, or 0 if none / on failure."""
        entries = tb.atlas_entries(self.picker_a.stem, self.picker_b.stem, names)
        if not entries or not self.results:
            return 0

        tile_w, tile_h = next(iter(self.results.values())).size
        atlas_path = os.path.join(folder, ATLAS_FILENAME)

        existing = None
        if os.path.isfile(atlas_path):
            try:
                existing = json.loads(Path(atlas_path).read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                errors.append(f"{ATLAS_FILENAME}: could not read existing file ({exc}); starting fresh")

        before = len(existing.get("tiles", [])) if existing else 0
        merged = tb.merge_atlas(existing, entries, tile_w, tile_h)
        added = len(merged["tiles"]) - before

        try:
            Path(atlas_path).write_text(json.dumps(merged, indent=2), encoding="utf-8")
        except OSError as exc:
            errors.append(f"{ATLAS_FILENAME}: {exc}")
            return 0
        return added

    # ------------------------------------------------------------ settings

    def _settings_dict(self) -> dict:
        return {
            "version": 1,
            "source_a": self.picker_a.path,
            "source_b": self.picker_b.path,
            "source_c": self.picker_c.path,
            "size": int(self.size_var.get()),
            "resample": self.resample_var.get(),
            "roughness": float(self.roughness_var.get()),
            "feather_pct": float(self.feather_var.get()),
            "blob_size": float(self.blob_var.get()),
            "transition_width_pct": float(self.transition_var.get()),
            "seed": int(self.seed_var.get()),
            "include_basic": bool(self.basic_var.get()),
            "include_diagonals": bool(self.diag_var.get()),
            "include_specials": bool(self.specials_var.get()),
            "include_border": bool(self.border_var.get()),
            "include_pure": bool(self.include_pure_var.get()),
            "isometric": bool(self.isometric_var.get()),
            "sequential": bool(self.sequential_var.get()),
            "start_index": int(self.start_index_var.get()),
            "output_format": self.format_var.get(),
            "output_folder": self.last_output_folder,
        }

    def _save_settings(self):
        try:
            SETTINGS_PATH.write_text(json.dumps(self._settings_dict(), indent=2), encoding="utf-8")
        except OSError:
            pass  # persistence is a convenience, not worth interrupting the user for

    def _load_settings(self):
        if not SETTINGS_PATH.exists():
            return
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return

        for path_attr, picker in (("source_a", self.picker_a), ("source_b", self.picker_b),
                                   ("source_c", self.picker_c)):
            path = data.get(path_attr)
            if path and os.path.isfile(path):
                picker.set_from_path(path)

        def restore(var, key, cast=lambda x: x):
            if key in data and data[key] is not None:
                try:
                    var.set(cast(data[key]))
                except (tk.TclError, ValueError, TypeError):
                    pass

        restore(self.size_var, "size", int)
        restore(self.resample_var, "resample", str)
        restore(self.roughness_var, "roughness", float)
        restore(self.feather_var, "feather_pct", float)
        restore(self.blob_var, "blob_size", float)
        restore(self.transition_var, "transition_width_pct", float)
        restore(self.seed_var, "seed", int)
        restore(self.basic_var, "include_basic", bool)
        restore(self.diag_var, "include_diagonals", bool)
        restore(self.specials_var, "include_specials", bool)
        restore(self.border_var, "include_border", bool)
        restore(self.include_pure_var, "include_pure", bool)
        restore(self.isometric_var, "isometric", bool)
        restore(self.sequential_var, "sequential", bool)
        restore(self.start_index_var, "start_index", int)
        restore(self.format_var, "output_format", str)

        self._toggle_sequential()
        self.last_output_folder = data.get("output_folder") or None
        self.status_var.set("Restored last-used settings.")


def main():
    app = TileGeneratorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
