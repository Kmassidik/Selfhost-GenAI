#!/usr/bin/env python3
"""'Neon Solitude' — a 2-min cinematic short. A lone woman in a rain-soaked neon
city at night, intercut with atmosphere. Same woman across her shots (consistent
description); pure-atmosphere b-roll fills between. 124 frames (~5s) each; run
through: upscale -> RIFE 60fps -> color grade -> edit to ~2 min + score.
Batch 1 = the first 8 (ns-face..ns-drops). Batch 2 = the next 8 (ns-eyes..ns-far)."""

_HER = ("a young woman with wet dark hair, pale skin, subtle freckles, melancholic dark eyes, "
        "wearing a black wool coat")
_CITY = ("a rain-soaked neon-lit city street at night, glowing magenta and cyan signs, "
         "wet reflections on the asphalt, soft rain falling, deep cinematic bokeh, fine film grain")
_LOOK = ("Cinematic live-action, anamorphic shallow depth of field, moody teal-and-magenta grade, "
         "low-key lighting. Slow, deliberate camera motion.")

def shot(desc, sound, music):
  return (f"""integrated_multimodal_description: [Shot 1] {_LOOK} {desc}

overall_soundscape: {sound}

non_diegetic_music: {music}""")

_RAIN = "Soft steady rain, distant city hum, faint traffic and neon buzz."
_SCORE = "A slow melancholic ambient score — low synth pad, a single piano note, deep and cinematic."

# ---- 'Kimi no Natsu' (Your Summer) — a Makoto Shinkai-style anime short ----
_GIRL = ("a teenage anime girl with short dark hair, a white-and-navy Japanese summer school uniform, "
         "large expressive eyes, soft blush, delicate features")
_ANIME = ("2D Japanese anime, cel-shaded hand-drawn animation, Makoto Shinkai style, lush painterly "
          "backgrounds, volumetric god rays, gentle lens flare, vivid saturated colors, highly detailed "
          "cumulus clouds, crisp line art. Smooth deliberate camera motion.")
_SUMMER = ("a nostalgic Japanese countryside summer afternoon, brilliant blue sky with towering white "
           "cumulus clouds, warm golden sunlight, cicadas, a quiet rural train station")
_ASOUND = "Summer cicadas, soft warm wind, a distant passing train, gentle natural ambience."
_AMUSIC = "A tender emotional piano melody with soft strings, nostalgic and bittersweet, anime film score."

# ---- 'SORAKIRI' (Sky-Cutter) — an epic anime trailer ----
_HERO = ("a young anime swordswoman, silver-white hair in a high ponytail, sharp determined amber eyes, "
         "a thin scar across one cheek, a tattered dark-blue battle cloak, holding a katana glowing faint blue")
_TRAILER = ("2D Japanese anime, cel-shaded hand-drawn animation, epic action-anime style like Ufotable/Demon Slayer, "
            "dramatic cinematic lighting, dynamic composition, highly detailed, vivid saturated colors, crisp line art, subtle film grain")
_TWORLD = ("a world of shattered floating islands and ancient ruins beneath a vast stormy sky, "
           "a spreading black void consuming the horizon")
_TSOUND = "Epic wind, distant rolling thunder, the faint ring of a drawn blade."
_TMUSIC = "An epic anime trailer score — thunderous taiko drums, a soaring choir, building orchestral strings, cinematic and heroic."

def tshot(desc, sound, music):  # trailer shot: NO live-action _LOOK prefix
  return (f"""integrated_multimodal_description: [Shot 1] {desc}

overall_soundscape: {sound}

non_diegetic_music: {music}""")

def ashot(desc):
  return shot(f"{_ANIME} {desc}", _ASOUND, _AMUSIC)

CENAS_H3 = {
  # === BATCH 1 — her (consistent subject) ===
  "ns-face":   {"image": None, "frames": 124, "prompt": shot(f"Tight close-up of {_HER}, rain beading on her skin, neon light washing across her face, in {_CITY}. She looks off-frame, then slowly to the lens. Slow push-in.", _RAIN, _SCORE)},
  "ns-walk":   {"image": None, "frames": 124, "prompt": shot(f"Medium tracking shot of {_HER} walking slowly toward camera down {_CITY}, coat catching the wind, neon smearing behind her. Camera glides backward.", _RAIN, _SCORE)},
  "ns-lookup": {"image": None, "frames": 124, "prompt": shot(f"Low-angle close-up of {_HER} lifting her face to the rain, eyes closing then opening, neon signs towering behind, in {_CITY}. Slow tilt up.", _RAIN, _SCORE)},
  "ns-profile":{"image": None, "frames": 124, "prompt": shot(f"Side-profile close-up of {_HER} standing still under a flickering neon sign in {_CITY}, breath faintly visible, rain streaking the frame. Locked, slow drift.", _RAIN, _SCORE)},
  # === BATCH 1 — atmosphere b-roll ===
  "ns-puddle": {"image": None, "frames": 124, "prompt": shot(f"Extreme close-up of a rain puddle on wet asphalt reflecting glowing magenta and cyan neon signs, ripples spreading as raindrops hit, in {_CITY}. Static, mesmerizing.", _RAIN, _SCORE)},
  "ns-sign":   {"image": None, "frames": 124, "prompt": shot(f"A close-up of a buzzing neon sign flickering in the rain, water running down glass, bokeh city lights behind, in {_CITY}. Slow rack focus.", _RAIN, _SCORE)},
  "ns-street": {"image": None, "frames": 124, "prompt": shot(f"A wide empty {_CITY}, a single figure small in the distance, neon reflections stretching down the wet road toward camera. Very slow push-in.", _RAIN, _SCORE)},
  "ns-drops":  {"image": None, "frames": 124, "prompt": shot(f"Extreme slow-motion close-up of rain droplets falling through colored neon light against black, glowing streaks, in {_CITY}. Static, abstract.", _RAIN, _SCORE)},

  # === BATCH 2 — her (consistent subject) ===
  "ns-eyes":   {"image": None, "frames": 124, "prompt": shot(f"Extreme close-up of the eyes of {_HER}, raindrops caught on her lashes, a single slow blink, neon reflected in her irises, in {_CITY}. Locked, intimate.", _RAIN, _SCORE)},
  "ns-window": {"image": None, "frames": 124, "prompt": shot(f"{_HER} seen as a soft reflection in a rain-streaked shop window, glowing neon signs behind the glass, her face overlaid with running water, in {_CITY}. Slow drift.", _RAIN, _SCORE)},
  "ns-back":   {"image": None, "frames": 124, "prompt": shot(f"Behind {_HER} as she walks slowly away from camera down {_CITY}, coat swaying, neon swallowing her silhouette, wet road glowing. Camera follows gently.", _RAIN, _SCORE)},
  "ns-turn":   {"image": None, "frames": 124, "prompt": shot(f"Three-quarter close-up of {_HER} slowly turning her head toward the lens, wet hair flicking a thin arc of water, neon bokeh behind, in {_CITY}. Slow motion.", _RAIN, _SCORE)},
  # === BATCH 2 — atmosphere b-roll ===
  "ns-hands":  {"image": None, "frames": 124, "prompt": shot(f"Close-up of the cupped hands of {_HER}, rain pooling in her palms, neon color shifting on her wet skin, in {_CITY}. Static, tender.", _RAIN, _SCORE)},
  "ns-cab":    {"image": None, "frames": 124, "prompt": shot(f"A lone taxi passing through {_CITY}, headlights sweeping across the wet road, tires throwing a fan of glowing spray, neon smearing in the motion. Slow pan.", _RAIN, _SCORE)},
  "ns-alley":  {"image": None, "frames": 124, "prompt": shot(f"A narrow neon alley in {_CITY}, steam rising from a street grate, colored signs receding into rain-haze, no one there. Very slow push-in.", _RAIN, _SCORE)},
  "ns-far":    {"image": None, "frames": 124, "prompt": shot(f"A very wide shot of {_HER} standing tiny beneath a huge glowing neon billboard, rain sheeting down through the light, in {_CITY}. Locked, epic and lonely.", _RAIN, _SCORE)},

  # === ANIME — 'Kimi no Natsu' — her (consistent subject). km-face renders FIRST as the style test. ===
  "km-face":    {"image": None, "frames": 124, "prompt": ashot(f"Emotional close-up of {_GIRL}, wind moving her hair, a soft wistful smile, sunlight flaring behind her, in {_SUMMER}. Slow push-in.")},
  "km-train":   {"image": None, "frames": 124, "prompt": ashot(f"{_GIRL} stands on the platform as a train rushes past, her hair and skirt blown by the wind, in {_SUMMER}. Camera holds, motion blur on the train.")},
  "km-look":    {"image": None, "frames": 124, "prompt": ashot(f"{_GIRL} slowly turns to look back over her shoulder toward the lens, eyes catching the light, in {_SUMMER}. Slow, tender.")},
  "km-walk":    {"image": None, "frames": 124, "prompt": ashot(f"Wide shot of {_GIRL} walking up a country road lined with swaying sunflowers, telephone poles receding, in {_SUMMER}. Camera tracks beside her.")},
  "km-rail":    {"image": None, "frames": 124, "prompt": ashot(f"{_GIRL} leans on a railing overlooking a glittering sea at golden hour, clouds glowing orange, wind in her hair. Slow drift.")},
  "km-window":  {"image": None, "frames": 124, "prompt": ashot(f"{_GIRL} rests her chin on her hand looking out a train window, her faint reflection over green countryside rushing past, warm light. Locked.")},
  "km-run":     {"image": None, "frames": 124, "prompt": ashot(f"{_GIRL} runs across an empty rural crossing as petals and light scatter around her, in {_SUMMER}. Camera tracks, joyful.")},
  "km-wave":    {"image": None, "frames": 124, "prompt": ashot(f"Backlit shot of {_GIRL} raising her hand to wave goodbye, sun blazing behind her into lens flare, in {_SUMMER}. Slow, bittersweet.")},
  # === ANIME — atmosphere (skies, trains, light — what these models render best) ===
  "km-clouds":  {"image": None, "frames": 124, "prompt": ashot("Towering white summer cumulus clouds drifting across a deep blue sky, sunlight breaking through in god rays, painterly and vast. Slow upward tilt.")},
  "km-station": {"image": None, "frames": 124, "prompt": ashot(f"An empty countryside train station platform bathed in warm afternoon light, long shadows, {_SUMMER}, no one there. Very slow push-in.")},
  "km-tracks":  {"image": None, "frames": 124, "prompt": ashot("Railroad tracks stretching straight to the horizon under a huge summer sky, heat haze shimmering, telephone wires overhead. Static, nostalgic.")},
  "km-field":   {"image": None, "frames": 124, "prompt": ashot("A vast field of sunflowers swaying in the wind under towering clouds, bright saturated greens and yellows. Slow lateral drift.")},
  "km-city":    {"image": None, "frames": 124, "prompt": ashot("A sweeping Tokyo cityscape at golden hour, trains crossing elevated tracks, sunlight glinting off thousands of windows, Shinkai style. Slow aerial push-in.")},
  "km-sky":     {"image": None, "frames": 124, "prompt": ashot("A vast pale-blue sky with a single white contrail and telephone wires crossing the frame, a lone bird, painterly clouds. Static, quiet, iconic.")},
  "km-rainwin": {"image": None, "frames": 124, "prompt": ashot("Sudden warm summer rain streaking down a window, blurred green garden and soft light beyond, droplets catching the sun. Static, intimate.")},
  "km-night":   {"image": None, "frames": 124, "prompt": ashot("A breathtaking starry night sky over dark countryside hills, the Milky Way glowing, a single shooting star crossing. Very slow drift.")},

  # === TRAILER TEST — look + dialogue lip-sync validation ===
  "tr-hero":     {"image": None, "frames": 124, "prompt": tshot(f"{_TRAILER} Heroic slow push-in on {_HERO}, standing on a cliff of broken stone in {_TWORLD}, cloak and hair whipping in the wind, she slowly raises her glowing katana toward the sky. Epic and cinematic.", _TSOUND, _TMUSIC)},
  "tr-heroline": {"image": None, "frames": 124, "prompt": tshot(f"{_TRAILER} Intense close-up of {_HERO} in {_TWORLD}. She looks straight into the lens and speaks, her lips moving in clear sync with her words, fierce and resolved.", "A young woman's clear, determined voice says aloud: \"I will cut down the sky itself.\" Then wind and distant thunder.", _TMUSIC)},
  # world / cold open
  "tr-sky":      {"image": None, "frames": 124, "prompt": tshot(f"{_TRAILER} A breathtaking wide establishing shot of {_TWORLD} at dawn, colossal broken islands drifting, golden light breaking through towering storm clouds. Slow majestic push-in.", _TSOUND, _TMUSIC)},
  "tr-void":     {"image": None, "frames": 124, "prompt": tshot(f"{_TRAILER} A creeping wall of black void devouring a landscape of ancient ruins, tendrils of darkness rising, an ominous red glow at its heart, in {_TWORLD}. Slow dread.", "Deep ominous rumble, cracking stone, a low unnatural hum.", _TMUSIC)},
  # hero
  "tr-heyes":    {"image": None, "frames": 124, "prompt": tshot(f"{_TRAILER} Extreme close-up of the determined amber eyes of {_HERO}, reflecting blue fire, unblinking resolve. Locked, intense.", _TSOUND, _TMUSIC)},
  "tr-draw":     {"image": None, "frames": 124, "prompt": tshot(f"{_TRAILER} Dynamic low-angle shot of {_HERO} drawing her katana, the blade igniting with blue sky-energy as it clears the sheath, sparks trailing, in {_TWORLD}. Dramatic.", "The sharp ring of a blade drawn, a surge of energy, wind.", _TMUSIC)},
  # allies / villain
  "tr-ally":     {"image": None, "frames": 124, "prompt": tshot(f"{_TRAILER} A young anime archer, a windswept teenage boy with messy brown hair and a green cloak, drawing a glowing bow atop a floating ruin in {_TWORLD}, confident smirk. Heroic.", _TSOUND, _TMUSIC)},
  "tr-villain":  {"image": None, "frames": 124, "prompt": tshot(f"{_TRAILER} A slow menacing reveal of the antagonist — a tall masked figure in flowing black armor, a single glowing red eye behind a cracked porcelain mask, standing before the void in {_TWORLD}. Ominous, cinematic.", "A low ominous drone, distant thunder.", _TMUSIC)},
  "tr-villainline":{"image": None, "frames": 124, "prompt": tshot(f"{_TRAILER} Intense close-up of the masked antagonist in {_TWORLD}, the mask's mouth moving as he speaks, cold and slow, red eye glowing.", "A deep, cold male voice says slowly: \"This world was already lost.\" Then a low ominous drone.", _TMUSIC)},
  # action montage
  "tr-run":      {"image": None, "frames": 124, "prompt": tshot(f"{_TRAILER} Dynamic tracking shot of {_HERO} sprinting across a crumbling stone bridge as it collapses behind her, debris flying, the void churning below, in {_TWORLD}. Fast, urgent.", "Crumbling stone, rushing wind, pounding footsteps.", _TMUSIC)},
  "tr-clash":    {"image": None, "frames": 124, "prompt": tshot(f"{_TRAILER} An explosive mid-air clash — the glowing blue katana of {_HERO} slamming against the antagonist's dark blade, a burst of sparks and a shockwave, in {_TWORLD}. Frozen power.", "A thunderous metallic clash, crackling energy.", _TMUSIC)},
  "tr-magic":    {"image": None, "frames": 124, "prompt": tshot(f"{_TRAILER} {_HERO} slashing the air and unleashing a massive arc of blue sky-energy that splits the storm clouds, a glowing magic sigil beneath her feet, in {_TWORLD}. Epic power.", "A roaring surge of energy, cracking thunder.", _TMUSIC)},
  "tr-dragon":   {"image": None, "frames": 124, "prompt": tshot(f"{_TRAILER} A colossal celestial sky-serpent of light and scales coiling through towering storm clouds, glowing eyes, high above {_TWORLD}. Awe-inspiring and vast.", "A deep resonant roar, rolling thunder, wind.", _TMUSIC)},
  "tr-leap":     {"image": None, "frames": 124, "prompt": tshot(f"{_TRAILER} Heroic wide shot of {_HERO} leaping off a shattered cliff toward the distant void, katana blazing blue, cloak streaming, tiny against the vast stormy sky of {_TWORLD}. Epic.", _TSOUND, _TMUSIC)},
  # emotional / climax
  "tr-tears":    {"image": None, "frames": 124, "prompt": tshot(f"{_TRAILER} An emotional close-up of {_HERO}, a single tear tracing her cheek, jaw set with grief and resolve, soft rim light, in {_TWORLD}. Quiet and moving.", "Soft wind, a single held piano note.", _TMUSIC)},
  "tr-duel":     {"image": None, "frames": 124, "prompt": tshot(f"{_TRAILER} A breathtaking wide silhouette of {_HERO} and the masked antagonist facing each other, blades raised, atop floating ruins above a sea of clouds at blazing sunset, in {_TWORLD}. Epic climax.", _TSOUND, _TMUSIC)},

  # === CP TEST — a real 15s (345-frame) continuous take @ 832x480; OOMs single-card, target for ring-2 CP ===
  "cp15":        {"image": None, "frames": 345, "prompt": tshot(f"{_TRAILER} A breathtaking continuous 15-second shot slowly pushing in across {_TWORLD} at dawn, colossal broken islands drifting through storm clouds, golden light sweeping over ancient ruins, {_HERO} standing tiny on a distant cliff. Smooth, majestic, unbroken camera move.", _TSOUND, _TMUSIC)},
  "cp10":        {"image": None, "frames": 243, "prompt": tshot(f"{_TRAILER} A breathtaking continuous 10-second shot slowly pushing in across {_TWORLD} at dawn, colossal broken islands drifting through storm clouds, golden light sweeping over ancient ruins, {_HERO} standing tiny on a distant cliff. Smooth, majestic, unbroken camera move.", _TSOUND, _TMUSIC)},
  # fallback: same 15s take at 768x448 (both dims /16 even) — lower activation, fits ring-2, upscale after
  "cp15lo":      {"image": None, "frames": 345, "height": 448, "width": 768, "prompt": tshot(f"{_TRAILER} A breathtaking continuous 15-second shot slowly pushing in across {_TWORLD} at dawn, colossal broken islands drifting through storm clouds, golden light sweeping over ancient ruins, {_HERO} standing tiny on a distant cliff. Smooth, majestic, unbroken camera move.", _TSOUND, _TMUSIC)},
  # OPTION 1 wall-test: single-card 15s at LOW res 512x288 (both /16-even) — the "long+soft" escape
  "cp15sd":      {"image": None, "frames": 345, "height": 288, "width": 512, "prompt": tshot(f"{_TRAILER} A breathtaking continuous 15-second shot slowly pushing in across {_TWORLD} at dawn, colossal broken islands drifting through storm clouds, golden light sweeping over ancient ruins, {_HERO} standing tiny on a distant cliff. Smooth, majestic, unbroken camera move.", _TSOUND, _TMUSIC)},
  # === 720p CAPABILITY TEST — native 1280x720 @ 124f (~5s); tokens ~0.83x the proven 15s@480p run, so the tiling recipe should fit ===
  "cp720":       {"image": None, "frames": 124, "height": 704, "width": 1280, "prompt": tshot(f"{_TRAILER} A breathtaking continuous shot slowly pushing in across {_TWORLD} at dawn, colossal broken islands drifting through storm clouds, golden light sweeping over ancient ruins, {_HERO} standing tiny on a distant cliff. Smooth, majestic, unbroken camera move.", _TSOUND, _TMUSIC)},

  # === ANIME LIP-SYNC TEST @ 720p (toward target.mp4) — a bespectacled man giving a passionate speech, lips in sync ===
  "an1":         {"image": None, "frames": 124, "height": 704, "width": 1280, "prompt": tshot(
      f"{_TRAILER} Medium close-up of a passionate young anime man with round glasses and messy dark hair, wearing a fur-collared winter coat, standing before a crowd in a city square under a bright overcast sky, tall buildings behind. He raises one fist and speaks fervently straight into the lens, his mouth moving in clear lip-sync with his words, brow furrowed with conviction. Slow cinematic push-in.",
      "A young man's earnest, impassioned voice says clearly, his lips moving in sync: \"This city is our home, and we are not leaving!\" Behind him a crowd murmurs and cheers, wind gusts through the square.",
      "A stirring, hopeful orchestral swell, cinematic and emotional.")},

  # === G4 SUMMIT: native 720p x 15s STRAIGHT (1280x704 x 345f ~= 93k tokens) — needs 3-GPU sequence parallelism ===
  "cp720long":   {"image": None, "frames": 345, "height": 704, "width": 1280, "prompt": tshot(f"{_TRAILER} A breathtaking continuous 15-second shot slowly pushing in across {_TWORLD} at dawn, colossal broken islands drifting through storm clouds, golden light sweeping over ancient ruins, {_HERO} standing tiny on a distant cliff. Smooth, majestic, unbroken camera move.", _TSOUND, _TMUSIC)},
}

if __name__ == "__main__":
  for k, v in CENAS_H3.items():
    print(f"== {k} (frames={v.get('frames', 124)})")
