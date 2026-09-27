"""Interactive map of expected graffiti reports in Zurich.

Run on the server with
    panel serve graffiti_app.py --setup graffiti_setup.py --port 5006 \
        --allow-websocket-origin=localhost:5006
(``--setup`` starts the initial fit at server startup instead of on the first request)
and forward the port with ``ssh -L 5006:localhost:5006 <host>``.
"""

import threading

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
from holoviews.streams import Pipe, RangeXY, Selection1D

from graffiti_model import CELL_SIZE_M, RateImage, get_job_manager
from utils import lonlat_to_mercator

hv.extension("bokeh")
pn.extension()


BASEMAPS = {
    "OpenStreetMap": "https://tile.openstreetmap.org/{Z}/{X}/{Y}.png",
    "swisstopo grey": "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/"
    "default/current/3857/{Z}/{X}/{Y}.jpeg",
    "swisstopo aerial": "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage/"
    "default/current/3857/{Z}/{X}/{Y}.jpeg",
}
DEBOUNCE_S = 0.3
HEAT_ALPHA = 0.6
# fixed color scale in expected reports per year per cell, higher values are clipped
HEAT_MAX = 25.0
# viridis: low values dark blue/purple, high values yellow
HEAT_CMAP = "viridis"
# area unit of the density, CELL_SIZE_M x CELL_SIZE_M
CELL_UNIT = f"{(CELL_SIZE_M / 1000) ** 2:g} km²"


@pn.cache
def get_reports() -> pd.DataFrame:
    gdf = get_job_manager().model.gdf
    x, y = lonlat_to_mercator(gdf.geometry.x.values, gdf.geometry.y.values)
    df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "date": pd.to_datetime(gdf["requested_datetime"]),
            "title": gdf["title"].fillna(""),
            "detail": gdf["detail"]
            .fillna("")
            .str.replace(r"\s*<br\s*/?>\s*", "\n", regex=True),
            "status": gdf["status"].fillna(""),
            "url": gdf["url"].fillna(""),
            "media_url": gdf["media_url"].fillna(""),
        }
    )
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d %H:%M")
    return df


class ReportList(pn.custom.JSComponent):
    """Chronological list of reports; ``selected`` scrolls the entry into view."""

    items = param.List(default=[])
    selected = param.Integer(default=-1)

    _esm = """
    export function render({ model, el }) {
      const list = document.createElement("div")
      list.className = "report-list"
      el.appendChild(list)

      function highlight(scroll) {
        for (const e of list.querySelectorAll(".selected")) e.classList.remove("selected")
        const entry = list.querySelector(`[data-id="${model.selected}"]`)
        if (!entry) return
        entry.classList.add("selected")
        // scroll only the list, scrollIntoView would also scroll the page
        if (scroll) list.scrollTo({
          top: entry.offsetTop - list.offsetTop - list.clientHeight / 2 + entry.clientHeight / 2,
          behavior: "smooth",
        })
      }

      function draw() {
        list.replaceChildren()
        const header = document.createElement("div")
        header.className = "count"
        header.textContent = `${model.items.length} reports in view`
        list.appendChild(header)
        for (const item of model.items) {
          const entry = document.createElement("div")
          entry.className = "entry"
          entry.dataset.id = item.id
          const img = item.media_url
            ? `<img src="${item.media_url}" loading="lazy">` : ""
          entry.innerHTML = `
            <div class="meta">${item.date_str} · ${item.status}</div>
            <div class="detail"></div>
            ${img}`
          const detail = entry.querySelector(".detail")
          detail.textContent = item.detail + " "
          const link = document.createElement("a")
          link.href = item.url
          link.target = "_blank"
          link.textContent = "report"
          detail.append(link)
          entry.addEventListener("click", (ev) => {
            if (ev.target.tagName !== "A") model.selected = item.id
          })
          list.appendChild(entry)
        }
        highlight(true)
      }

      model.on("items", draw)
      model.on("selected", () => highlight(true))
      draw()
    }
    """

    _stylesheets = [
        """
        .report-list {
          height: calc(100vh - 260px); overflow-y: auto; font-size: 13px;
          position: relative;
        }
        .count { font-weight: 600; padding: 4px 6px; }
        .entry { padding: 6px; border-bottom: 1px solid #ddd; cursor: pointer; }
        .entry:hover { background: #f4f4f4; }
        .entry.selected { background: #ffe8c2; }
        .meta { color: #666; font-size: 11px; }
        .detail { white-space: pre-wrap; }
        .entry img { max-width: 100%; max-height: 160px; margin-top: 4px; }
        """
    ]


def create_app():
    jm = get_job_manager()
    reports = get_reports()
    doc = pn.state.curdoc
    bx0, by0, bx1, by1 = jm.model.mercator_bounds

    # --- widgets ---
    lock = pn.widgets.Switch(name="Lock view", value=False)
    progress = pn.indicators.Progress(value=0, max=100, sizing_mode="stretch_width")
    status = pn.pane.Str("")
    report_list = ReportList(sizing_mode="stretch_both", min_height=400)

    # --- map ---
    basemap = pn.widgets.Select(
        name="Base map", options=list(BASEMAPS), value="OpenStreetMap"
    )
    tiles = hv.DynamicMap(
        pn.bind(lambda name: hv.Tiles(BASEMAPS[name], name=name), basemap)
    )

    def empty_image():
        return {
            "x0": bx0,
            "y0": by0,
            "x1": bx1,
            "y1": by1,
            "Z": np.full((2, 2), np.nan),
        }

    image_pipe = Pipe(data=empty_image())
    # initial city-wide density, shown where the view density does not reach
    base_pipe = Pipe(data=empty_image())
    initial = {"result": None}

    def make_image(data, colorbar=False):
        # the bounds are the pixel edges; Z rows run along increasing y but hv.Image
        # expects the first row at the top
        img = hv.Image(
            data["Z"][::-1],
            bounds=(data["x0"], data["y0"], data["x1"], data["y1"]),
            kdims=["x", "y"],
            vdims=[hv.Dimension("rate", label=f"reports/yr per {CELL_UNIT}")],
        )
        return img.opts(
            cmap=HEAT_CMAP,
            clim=(0, HEAT_MAX),
            alpha=HEAT_ALPHA,
            clipping_colors={"NaN": "transparent"},
            colorbar=colorbar,
            colorbar_opts={"title": f"expected reports / year / {CELL_UNIT}"},
            tools=["hover"],
        )

    heat = hv.DynamicMap(make_image, streams=[image_pipe])
    base_heat = hv.DynamicMap(
        lambda data: make_image(data, colorbar=True), streams=[base_pipe]
    )

    points = hv.Points(reports, kdims=["x", "y"], vdims=["date_str", "title"]).opts(
        marker="x",
        color="#e00000",
        size=9,
        line_width=3,
        alpha=0.9,
        tools=["tap", "hover"],
        nonselection_alpha=0.9,
        selection_color="#e00000",
        hover_tooltips=[("date", "@date_str"), ("title", "@title")],
    )
    selection = Selection1D(source=points)

    highlight_pipe = Pipe(data=reports.iloc[:0])
    # selected report: thick cyan ring with a black outline, drawn above the crosses
    highlight_outline = hv.DynamicMap(
        lambda data: hv.Points(data, kdims=["x", "y"]), streams=[highlight_pipe]
    ).opts(marker="circle", size=34, fill_alpha=0, line_color="black", line_width=8)
    highlight = hv.DynamicMap(
        lambda data: hv.Points(data, kdims=["x", "y"]), streams=[highlight_pipe]
    ).opts(
        marker="circle",
        size=34,
        fill_color="#00e5ff",
        fill_alpha=0.25,
        line_color="#00e5ff",
        line_width=4,
    )

    range_stream = RangeXY(source=points)

    overlay = (tiles * base_heat * heat * points * highlight_outline * highlight).opts(
        xlim=(bx0, bx1),
        ylim=(by0, by1),
        xaxis=None,
        yaxis=None,
        active_tools=["wheel_zoom", "pan", "tap"],
        responsive=True,
        min_height=600,
    )

    # --- view -> list and refits ---
    state = {"view": (bx0, by0, bx1, by1), "timer": None}

    def current_view():
        return state["view"]

    def update_list():
        x0, y0, x1, y1 = current_view()
        in_view = reports[
            (reports.x >= x0)
            & (reports.x <= x1)
            & (reports.y >= y0)
            & (reports.y <= y1)
        ].sort_values("date")
        report_list.items = [
            {"id": int(i), **row}
            for i, row in in_view[
                ["date_str", "detail", "status", "url", "media_url"]
            ].iterrows()
        ]

    def submit_refit():
        if not lock.value:
            jm.submit(current_view())

    def on_range(x_range, y_range):
        if x_range is None or y_range is None:
            return
        state["view"] = (x_range[0], y_range[0], x_range[1], y_range[1])
        # debounce: only act once the view stopped changing
        if state["timer"] is not None:
            state["timer"].cancel()

        def fire():
            doc.add_next_tick_callback(update_list)
            submit_refit()

        state["timer"] = threading.Timer(DEBOUNCE_S, fire)
        state["timer"].start()

    range_stream.add_subscriber(on_range)
    lock.param.watch(lambda e: None if e.new else submit_refit(), "value")

    # --- selection sync ---
    def on_tap(index):
        report_list.selected = int(reports.index[index[0]]) if index else -1

    selection.add_subscriber(on_tap)

    def on_list_select(event):
        highlight_pipe.send(
            reports.loc[[event.new]] if event.new in reports.index else reports.iloc[:0]
        )

    report_list.param.watch(on_list_select, "selected")

    # --- job manager -> UI (called from the worker thread) ---
    def on_progress(job_id, fraction, stage):
        def apply():
            progress.value = int(100 * fraction)
            status.object = f"{stage} ({100 * fraction:.0f}%)"

        doc.add_next_tick_callback(apply)

    def image_data(result: RateImage) -> dict:
        return dict(x0=result.x0, y0=result.y0, x1=result.x1, y1=result.y1, Z=result.Z)

    def base_data(view: RateImage | None) -> dict:
        """Initial density with the area of the view density blanked out, both are
        semi-transparent and would otherwise add up where they overlap."""
        data = image_data(initial["result"])
        if view is not None:
            # the view image is snapped to the pixel edges of the initial image
            ny, nx = data["Z"].shape
            px = (data["x1"] - data["x0"]) / nx
            py = (data["y1"] - data["y0"]) / ny
            i0, i1 = (round((x - data["x0"]) / px) for x in (view.x0, view.x1))
            j0, j1 = (round((y - data["y0"]) / py) for y in (view.y0, view.y1))
            data["Z"] = data["Z"].copy()
            data["Z"][j0:j1, i0:i1] = np.nan
        return data

    def on_result(result: RateImage):
        if result.job_id == 0:
            initial["result"] = result
            if jm.last_result is None:
                doc.add_next_tick_callback(lambda: base_pipe.send(base_data(None)))
            return

        def apply():
            image_pipe.send(image_data(result))
            if initial["result"] is not None:
                base_pipe.send(base_data(result))

        doc.add_next_tick_callback(apply)

    jm.subscribe(on_progress, on_result)
    pn.state.on_session_destroyed(lambda ctx: jm.unsubscribe(on_progress, on_result))

    # show what is known already, then refit the initial view once the model is ready
    if jm.initial_result is not None:
        on_result(jm.initial_result)
    if jm.last_result is not None:
        on_result(jm.last_result)
    on_progress(*jm.progress)
    update_list()

    # queued behind the initial global fit on the worker
    pn.state.onload(submit_refit)

    return pn.template.FastListTemplate(
        title="Züri wie neu: graffiti reports",
        sidebar=[lock, basemap, progress, status, report_list],
        sidebar_width=380,
        main=[pn.pane.HoloViews(overlay, sizing_mode="stretch_both")],
    )


create_app().servable()
