import datetime

import numpy as np
from bs4 import BeautifulSoup

with open("stlite/index.html", "r") as f:
    soup = BeautifulSoup(f, "html.parser")

with open("stlite/metapop_stlite.zip", "rb") as f:
    zip = f.read()

stlite_js1 = """
      import { mount } from "https://cdn.jsdelivr.net/npm/@stlite/browser@0.81.6/build/stlite.js";
      mount(
        {
            requirements: ["tabulate", "pandas", "altair", "numpy", "st_flexible_callout_elements"],
            entrypoint: "launch_stlite.py",
            archives: [
                {
                    buffer: new Uint8Array(["""

stlite_js2 = """]),
                    format: "zip",
                    options: {},
                },
            ],
        },
        document.getElementById("root")
    );
"""

jselem = soup.find("script", {"type": "module"})
jselem.string.replace_with(
    stlite_js1
    + ",".join(f"0x{n:x}" for n in np.frombuffer(zip, dtype=np.uint8))
    + stlite_js2
)

foot = soup.find("div", {"id": "foot"})
foot.string.replace_with(
    datetime.datetime.now(datetime.timezone.utc).strftime(
        "Please be patient as the simulator loads.  This version built on %Y-%m-%d at %H:%M:%S UTC."
    )
)

with open("stlite/measles_sim.html", "w") as f:
    f.write(str(soup))
