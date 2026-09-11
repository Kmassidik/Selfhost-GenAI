# Scenes

`h3_cenas.py` is ours — the scene definitions (prompts, resolutions, frame
counts) used by every render in this project.

It is consumed by the **h3-consumer-bench** engine, which is third-party:

    https://github.com/sztlink/h3-consumer-bench

To reproduce a render:

    git clone https://github.com/sztlink/h3-consumer-bench.git runtime/h3-consumer-bench
    cp scenes/h3_cenas.py runtime/h3-consumer-bench/h3_cenas.py

Then follow the phase A / phase B instructions in the top-level README.
