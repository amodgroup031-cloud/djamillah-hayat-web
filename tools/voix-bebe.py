"""Voix naturelle de DJAMILLAH HAYAT, vérifiée à l'oreille par reconnaissance vocale.
- Français : Kokoro (voix « siwis », très naturelle), sinon Piper.
- Arabe : Piper « kareem » (voix arabe native), lettres écrites avec toutes leurs voyelles.
- Chaque enregistrement est réécouté par Whisper : s'il n'est pas compris correctement, il est refait
  avec d'autres réglages ; s'il échoue encore, il est retiré (l'application lit alors avec la voix du téléphone).
- Les versets du Coran ne sont jamais synthétisés : ils sont récités par Al-Husary.
Sorties : www/voix/<clé>.mp3, www/voix/index.json, www/voix/qualite.json, res-voix/*.ogg   © AMOD GROUP"""
import json, os, re, subprocess, sys, unicodedata, urllib.request, wave
import numpy as np
PHASE = sys.argv[1] if len(sys.argv) > 1 else "all"           # fr | ar | all
PART, NPARTS = (int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0/1").split("/"))
OUT = f"sortie-{PHASE}-{PART}"

os.makedirs("modeles", exist_ok=True); os.makedirs(f"{OUT}/voix", exist_ok=True); os.makedirs(f"{OUT}/res-voix", exist_ok=True); os.makedirs("tmp", exist_ok=True)
BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
PIPER = {"ar": ["ar/ar_JO/kareem/medium/ar_JO-kareem-medium", "ar/ar_JO/kareem/low/ar_JO-kareem-low"],
         "fr": ["fr/fr_FR/siwis/medium/fr_FR-siwis-medium", "fr/fr_FR/tom/medium/fr_FR-tom-medium"]}

# ---------- Moteurs ----------
from piper.voice import PiperVoice
def load_piper(lang):
    for m in PIPER[lang]:
        name = m.split("/")[-1]
        try:
            for ext in (".onnx", ".onnx.json"):
                dst = f"modeles/{name}{ext}"
                if not os.path.exists(dst): urllib.request.urlretrieve(BASE + m + ext, dst)
            print("Piper", lang, name); return PiperVoice.load(f"modeles/{name}.onnx", config_path=f"modeles/{name}.onnx.json")
        except Exception as e: print("indisponible", name, e)
    return None
PV = {l: (load_piper(l) if PHASE in (l, "all") else None) for l in PIPER}

KOKORO = None
try:
    if PHASE not in ("fr", "all"): raise RuntimeError("phase arabe")
    from kokoro import KPipeline
    KOKORO = KPipeline(lang_code="f", repo_id="hexgrad/Kokoro-82M")
    print("Kokoro prêt (français)")
except Exception as e:
    print("Kokoro indisponible :", e)

CB = None
try:
    if PHASE not in ("ar", "all"): raise RuntimeError("phase française")
    import torchaudio as ta
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    CB = ChatterboxMultilingualTTS.from_pretrained(device="cpu"); print("Chatterbox prêt (arabe)")
except Exception as e:
    print("Chatterbox indisponible :", e)
def synth_cb(text, out):
    wv = CB.generate(text, language_id="ar"); ta.save(out, wv, CB.sr)

ASR = None
ASR_AR = None
try:
    if PHASE not in ("ar", "all"): raise RuntimeError("phase française")
    from faster_whisper import WhisperModel as _WM
    ASR_AR = _WM("large-v3", device="cpu", compute_type="int8"); print("Whisper large prêt (arabe)")
except Exception as e:
    print("Whisper large indisponible :", e)
try:
    if PHASE not in ("fr", "all"): raise RuntimeError("phase arabe")
    from faster_whisper import WhisperModel
    ASR = WhisperModel("small", device="cpu", compute_type="int8")
    print("Whisper prêt")
except Exception as e:
    print("Whisper indisponible :", e)

# ---------- Prononciation ----------
# Noms des lettres écrits avec toutes leurs voyelles, pour une prononciation exacte
NOMS = {"ألف":"أَلِفْ","باء":"بَاءْ","تاء":"تَاءْ","ثاء":"ثَاءْ","جيم":"جِيمْ","حاء":"حَاءْ","خاء":"خَاءْ","دال":"دَالْ","ذال":"ذَالْ","راء":"رَاءْ","زاي":"زَايْ","سين":"سِينْ","شين":"شِينْ","صاد":"صَادْ","ضاد":"ضَادْ","طاء":"طَاءْ","ظاء":"ظَاءْ","عين":"عَيْنْ","غين":"غَيْنْ","فاء":"فَاءْ","قاف":"قَافْ","كاف":"كَافْ","لام":"لَامْ","ميم":"مِيمْ","نون":"نُونْ","هاء":"هَاءْ","واو":"وَاوْ","ياء":"يَاءْ"}
FIX_FR = {"Allâhoumma": "Allahoumma", "Allah": "Allâh", "souhour": "souhoûr", "Fajr": "Fadjr", "'Asr": "Asre", "Asr": "Asre", "Dohr": "Dohre", "Dhohr": "Dohre",
          "Djamila Hayat": "Djamila Hayatt", "DJAMILLAH HAYAT": "Djamila Hayatt", "bil-khayr": "bil khaïr", "'Ichâ": "Icha", "tajwîd": "tadjouid"}
def prep(l, t):
    t = t.strip()
    if l == "fr":
        t = t.replace("ﷺ", ", paix et salut sur lui,").replace("«", "").replace("»", "").replace("’", "'")
        t = re.sub(r"\(([^)]*\d[^)]*)\)", "", t)            # références entre parenthèses (Bukhari 527…) non lues
        t = re.sub(r"\s+([,.;:!?])", r"\1", t)
        for a, b in FIX_FR.items(): t = re.sub(r"\b" + re.escape(a) + r"\b", b, t)
    else:
        t = NOMS.get(t, t).replace("ﷺ", " صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ ")
    return " ".join(t.split())

def save_wav(path, audio, sr):
    a = np.clip(np.asarray(audio, dtype=np.float32), -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((a * 32767).astype(np.int16).tobytes())

def synth_piper(l, text, out, ls=1.0, ns=0.5, nw=0.6):
    with wave.open(out, "wb") as w:
        PV[l].synthesize(text, w, length_scale=ls, noise_scale=ns, noise_w=nw, sentence_silence=0.3)

def synth_kokoro(text, out, speed=0.95):
    chunks = [a for _, _, a in KOKORO(text, voice="ff_siwis", speed=speed, split_pattern=r"(?<=[.!?;])\s+")]
    gap = np.zeros(int(24000 * 0.18), dtype=np.float32)
    audio = np.concatenate([np.concatenate([np.asarray(c, dtype=np.float32), gap]) for c in chunks])
    save_wav(out, audio, 24000)

# ---------- Vérification à l'oreille ----------
MARKS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")
def norm(l, s):
    s = unicodedata.normalize("NFKC", s.lower())
    if l == "ar":
        s = MARKS.sub("", s).replace("ٱ", "ا").replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي").replace("ء", "")
        s = re.sub(r"[^ء-ي]", "", s)
    else:
        s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
        s = re.sub(r"[^a-z0-9]", "", s)
    return s
def sim(a, b):
    if not a or not b: return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1 - prev[-1] / max(len(a), len(b))
def heard(l, wav, hint=None):
    segs, _ = (ASR_AR if (l == "ar" and ASR_AR) else ASR).transcribe(wav, language=l, beam_size=5, vad_filter=False, condition_on_previous_text=False, initial_prompt=hint)
    return "".join(s.text for s in segs).strip()
def check(l, text, wav):
    """Score de 0 à 1 : ce que l'oreille entend comparé au texte attendu."""
    if not (ASR_AR if l == "ar" else ASR): return 0.0, "pas de vérification"
    ref = norm(l, NOMS.get(text, text) if l == "ar" else prep(l, text))
    if l == "ar" and len(ref) < 3: return None, ""        # syllabe isolée : la reconnaissance vocale ne peut pas juger
    h = heard(l, wav)
    return sim(ref, norm(l, h)), h

def encode(wav, out, fmt="mp3"):
    args = ["-c:a", "libmp3lame", "-b:a", "40k"] if fmt == "mp3" else ["-c:a", "libvorbis", "-q:a", "4"]
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", wav, "-af",
                    "silenceremove=start_periods=1:start_threshold=-50dB,areverse,silenceremove=start_periods=1:start_threshold=-50dB,areverse,loudnorm=I=-16:TP=-1.5:LRA=11,apad=pad_dur=0.12",
                    "-ac", "1", "-ar", "24000", *args, out], check=True)

SEUIL = {"fr": 0.88, "ar": 0.88}
def produce(l, text, slow, solo=False):
    """Essaie plusieurs voix/réglages, garde le meilleur résultat vérifié."""
    t = prep(l, text); tries = []
    if l == "fr":
        if KOKORO: tries += [("kokoro", lambda o: synth_kokoro(t, o, 0.95)), ("kokoro-lent", lambda o: synth_kokoro(t, o, 0.85))]
        if PV["fr"]: tries += [("piper", lambda o: synth_piper("fr", t, o, 1.0))]
    else:
        if not solo and (slow or len(t.split()) < 2): return (None, "isole", 0, "")
        if CB: tries += [("chatterbox", lambda o: synth_cb(t, o))]
        base = 1.05
        tries += [("piper", lambda o: synth_piper("ar", t, o, base)), ("piper-lent", lambda o: synth_piper("ar", t, o, base + 0.2, 0.4, 0.5)),
                  ("piper-clair", lambda o: synth_piper("ar", t, o, base + 0.1, 0.3, 0.3))]
    best = None
    for name, fn in tries:
        wav = f"tmp/{name}.wav"
        try: fn(wav)
        except Exception as e: print("erreur", name, e); continue
        sc, h = check(l, text, wav)
        if sc is None: return wav, name, None, ""
        if not best or sc > best[2]: best = (wav, name, sc, h); os.replace(wav, "tmp/best.wav"); best = ("tmp/best.wav", name, sc, h)
        if sc >= 0.97: break
    return best if best else (None, None, 0, "")

liste = json.load(open(os.environ.get("LISTE", "scripts/voix-liste.json"), encoding="utf8"))
liste = [x for x in liste if not x.get("q") and (PHASE == "all" or x["l"] == PHASE)]
liste = liste[PART::NPARTS]
print("à fabriquer :", len(liste), flush=True)
index, qualite = {}, {"ok": 0, "retires": [], "non_verifies": 0, "moteurs": {}}
for i, x in enumerate(liste):
    if x.get("q"): index[x["k"]] = "q:" + x["q"]; continue
    wav, eng, sc, h = produce(x["l"], x["t"], bool(x.get("slow")), bool(x.get("solo")))
    if eng == "isole": qualite.setdefault("isoles", 0); qualite["isoles"] += 1; continue
    if not wav: qualite["retires"].append({"t": x["t"][:60], "raison": "échec"}); continue
    if sc is not None and sc < (0.5 if x.get("name") else SEUIL[x["l"]]):
        qualite["retires"].append({"t": x["t"][:60], "entendu": h[:60], "score": round(sc, 2)}); continue
    encode(wav, f"{OUT}/voix/{x['k']}.mp3"); index[x["k"]] = 1
    qualite["ok"] += 1; qualite["moteurs"][eng] = qualite["moteurs"].get(eng, 0) + 1
    if sc is None: qualite["non_verifies"] += 1
    print(i + 1, "/", len(liste), eng, None if sc is None else round(sc, 2), flush=True)
json.dump(index, open(f"{OUT}/index.json", "w"), separators=(",", ":"))

# ---------- Messages parlés des notifications ----------
PR = {"fajr": ("Fajr", "الْفَجْرِ"), "dhuhr": ("Dohr", "الظُّهْرِ"), "asr": ("Asr", "الْعَصْرِ"), "maghrib": ("Maghreb", "الْمَغْرِبِ"), "isha": ("Icha", "الْعِشَاءِ")}
CLIPS = {
 "fr": {"matin": "Assalamou alaykoum ! C'est l'heure de votre programme du jour. Ouvrez Djamila Hayat : je vais vous dire ce qu'il faut faire aujourd'hui.",
        "mort": "Souviens-toi de la mort. Elle peut venir aujourd'hui. Prépare-toi : prie à l'heure, demande pardon, rends ce que tu dois. Et ne désespère jamais de la miséricorde d'Allah.",
        "plan": "C'est l'heure de faire le point sur votre plan spirituel. Avez-vous accompli vos objectifs aujourd'hui ?",
        "jeune": "Demain est un jour de jeûne. Pensez au repas de l'aube. Qu'Allah vous facilite.",
        "tawba": "Encore un pas vers Allah aujourd'hui ? Prenez une minute pour noter votre journée.",
        **{k: f"C'est l'heure de la prière du {v[0]}. Venez à la prière, venez à la réussite." for k, v in PR.items()}},
 "ar": {"matin": "السَّلَامُ عَلَيْكُمْ! حَانَ وَقْتُ بَرْنَامَجِكَ الْيَوْمِيِّ. اِفْتَحِ التَّطْبِيقَ لِأُخْبِرَكَ بِمَا تَفْعَلُهُ الْيَوْمَ.",
        "mort": "تَذَكَّرِ الْمَوْتَ، فَقَدْ يَأْتِي الْيَوْمَ. اِسْتَعِدَّ لَهُ: صَلِّ فِي الْوَقْتِ، وَاسْتَغْفِرْ، وَرُدَّ الْحُقُوقَ. وَلَا تَيْأَسْ أَبَدًا مِنْ رَحْمَةِ اللَّهِ.",
        "plan": "حَانَ وَقْتُ مُرَاجَعَةِ خُطَّتِكَ الْإِيمَانِيَّةِ. هَلْ أَنْجَزْتَ أَهْدَافَكَ الْيَوْمَ؟",
        "jeune": "غَدًا يَوْمُ صِيَامٍ إِنْ شَاءَ اللَّهُ، فَلَا تَنْسَ السَّحُورَ.",
        "tawba": "خُطْوَةٌ أُخْرَى نَحْوَ اللَّهِ الْيَوْمَ؟ خُذْ دَقِيقَةً لِتُسَجِّلَ يَوْمَكَ.",
        **{k: f"حَانَ الْآنَ وَقْتُ صَلَاةِ {v[1]}. حَيَّ عَلَى الصَّلَاةِ، حَيَّ عَلَى الْفَلَاحِ." for k, v in PR.items()}}}
for l, clips in CLIPS.items():
    if os.environ.get("LISTE"): break
    if PART != 0 or PHASE not in (l, "all"): continue
    for k, txt in clips.items():
        wav, eng, sc, h = produce(l, txt, False)
        if not wav or sc is None or sc < 0.93:
            qualite.setdefault("notifications", {})[f"{k}_{l}"] = ["refusé (carillon à la place)", None if sc is None else round(sc, 2), (h or "")[:80]]; continue
        encode(wav, f"{OUT}/res-voix/djh_{k}_{l}.ogg", "ogg")
        qualite.setdefault("notifications", {})[f"{k}_{l}"] = [eng, None if sc is None else round(sc, 2), h[:80]]
json.dump(qualite, open(f"{OUT}/qualite.json", "w"), ensure_ascii=False, indent=1)
print("RÉSUMÉ voix :", qualite["ok"], "fichiers ; retirés :", len(qualite["retires"]), "; lettres/syllabes/mots isolés laissés à la voix du téléphone ou de l'oustaz :", qualite.get("isoles", 0), "; moteurs :", qualite["moteurs"])
for r in qualite["retires"][:40]: print("RETIRÉ", r)
for k, v in qualite.get("notifications", {}).items(): print("NOTIF", k, v)
