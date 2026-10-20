// Scroll-scrubbed render sequences + telemetry readout.
(() => {
  const root = document.documentElement;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const mobile = matchMedia('(max-width: 720px)').matches;
  const variant = mobile ? 'm' : 'd';
  const step = mobile ? 2 : 1;          // phones get every other frame
  const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
  const smooth = t => t * t * (3 - 2 * t);

  // ---------- frame sequences ----------
  const seqs = {};
  function sequence(name, total) {
    if (seqs[name]) return seqs[name];
    const idx = [];
    for (let i = 0; i < total; i += step) idx.push(i);
    const s = { name, idx, imgs: new Array(idx.length), loaded: new Uint8Array(idx.length), started: false, onload: [] };
    s.src = i => `assets/frames/${name}/${variant}/${String(idx[i]).padStart(4, '0')}.webp`;
    s.load = () => {
      if (s.started) return; s.started = true;
      // first, last, then coarse-to-fine so scrubbing works before everything arrives
      const order = [0, idx.length - 1];
      for (let stride = 16; stride >= 1; stride >>= 1)
        for (let i = 0; i < idx.length; i += stride) if (!order.includes(i)) order.push(i);
      let inflight = 0, k = 0;
      const pump = () => {
        while (inflight < 6 && k < order.length) {
          const i = order[k++]; inflight++;
          const img = new Image(); img.decoding = 'async'; img.src = s.src(i);
          img.onload = () => { s.imgs[i] = img; s.loaded[i] = 1; inflight--; s.onload.forEach(f => f(i)); pump(); };
          img.onerror = () => { inflight--; pump(); };
        }
      };
      pump();
    };
    s.nearest = i => {
      for (let d = 0; d < idx.length; d++) {
        if (s.loaded[i - d]) return s.imgs[i - d];
        if (s.loaded[i + d]) return s.imgs[i + d];
      }
      return null;
    };
    return (seqs[name] = s);
  }

  // canvas sizes come from a ResizeObserver, so the render loop never reads layout
  const sizes = new WeakMap();
  const ro = new ResizeObserver(entries => { for (const e of entries) sizes.set(e.target, e.contentRect); dirty = true; });
  function fitCanvas(c) {
    if (!sizes.has(c)) { ro.observe(c); sizes.set(c, c.getBoundingClientRect()); }
    const box = sizes.get(c), dpr = Math.min(devicePixelRatio || 1, 2);
    const w = Math.round(box.width * dpr), h = Math.round(box.height * dpr);
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; c._last = null; }
  }
  function draw(c, img) {
    if (!img || c._last === img) return;
    const ctx = c.getContext('2d');
    const cw = c.width, ch = c.height, r = Math.max(cw / img.naturalWidth, ch / img.naturalHeight);
    const w = img.naturalWidth * r, h = img.naturalHeight * r;
    ctx.drawImage(img, (cw - w) / 2, (ch - h) / 2, w, h);
    c._last = img;
  }

  // ---------- sections ----------
  const hero = document.querySelector('.hero');
  const bodies = document.querySelector('.bodies');
  const heroCanvas = hero.querySelector('.seq');
  const [bodyA, bodyB] = bodies.querySelectorAll('.seq');
  const heroCopy = hero.querySelector('.hero-copy');
  const cue = hero.querySelector('.scroll-cue');
  const humanoid = sequence('humanoid', +hero.dataset.frames);
  const quadruped = sequence('quadruped', +bodies.dataset.frames);
  const arm = sequence('arm', +bodies.dataset.frames);
  const bodyItems = [...bodies.querySelectorAll('.body-list li')];
  const rail = bodies.querySelector('.body-rail i');
  const nav = document.querySelector('.nav');

  humanoid.onload.push(i => { if (i === 0) { requestAnimationFrame(() => root.classList.add('ready')); quadruped.load(); arm.load(); } dirty = true; });
  quadruped.onload.push(() => (dirty = true));
  arm.onload.push(() => (dirty = true));
  humanoid.load();
  // reveal the page even if frames are slow to arrive
  setTimeout(() => root.classList.add('ready'), 1800);

  // telemetry: joint angles exported from the same MuJoCo motion that drove the render
  const tele = { data: null, cells: [...hero.querySelectorAll('[data-j]')], clock: hero.querySelector('[data-clock]') };
  fetch('assets/frames/humanoid/telemetry.json').then(r => r.json()).then(d => { tele.data = d; dirty = true; }).catch(() => {});

  const progress = el => {
    const r = el.getBoundingClientRect();
    return clamp(-r.top / (r.height - innerHeight));
  };

  // Scroll progress per stage. GSAP ScrollTrigger scrubs it with a short catch-up, so the frames
  // glide to rest after the wheel stops instead of stepping. Without GSAP, read the rects directly.
  const st = { hero: 0, bodies: 0 };
  const pins = [];
  const hasGsap = !!(window.gsap && window.ScrollTrigger);
  if (hasGsap) {
    gsap.registerPlugin(ScrollTrigger);
    for (const [key, el] of [['hero', hero], ['bodies', bodies]]) {
      gsap.to(st, {
        [key]: 1, ease: 'none', onUpdate: () => (dirty = true),
        scrollTrigger: { trigger: el, start: 'top top', end: 'bottom bottom', scrub: reduced ? true : 0.45 },
      });
      pins.push(ScrollTrigger.create({ trigger: el, start: 'top top', end: 'bottom bottom', onToggle: () => (dirty = true) }));
    }
  }

  let dirty = true, lastFrame = -1, bodyOn = 0;
  function render() {
    if (!hasGsap) { st.hero = progress(hero); st.bodies = progress(bodies); }
    // hero
    const ph = st.hero;
    const fh = Math.round(ph * (humanoid.idx.length - 1));
    fitCanvas(heroCanvas);
    draw(heroCanvas, humanoid.nearest(fh));
    const fade = smooth(clamp((ph - 0.42) / 0.3));
    heroCopy.style.opacity = 1 - fade;
    heroCopy.style.transform = `translateY(${-fade * 40}px)`;
    heroCopy.style.visibility = fade >= 1 ? 'hidden' : '';
    cue.style.opacity = 1 - clamp(ph * 8);
    cue.style.animationPlayState = ph > 0.13 ? 'paused' : 'running';
    if (tele.data && fh !== lastFrame) {
      const src = humanoid.idx[fh];
      tele.cells.forEach(dd => { const v = tele.data.joints[dd.dataset.j]?.[src]; if (v != null) dd.textContent = (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2); });
      tele.clock.textContent = `t ${(src / tele.data.fps).toFixed(2)} s`;
      lastFrame = fh;
    }

    // bodies: first half quadruped, second half arm, crossfade in the middle
    const pb = st.bodies;
    const a = clamp(pb / 0.5), b = clamp((pb - 0.5) / 0.5);
    fitCanvas(bodyA); fitCanvas(bodyB);
    draw(bodyA, quadruped.nearest(Math.round(a * (quadruped.idx.length - 1))));
    draw(bodyB, arm.nearest(Math.round(b * (arm.idx.length - 1))));
    // tight crossfade bridged by a brief blur, so the two scenes never sit in a double exposure
    const x = clamp((pb - 0.47) / 0.06), haze = Math.sin(Math.PI * x);
    bodyB.style.opacity = smooth(x);
    const blur = reduced || haze < 0.01 ? '' : `blur(${(haze * 10).toFixed(1)}px)`;
    if (bodyA.style.filter !== blur) bodyA.style.filter = bodyB.style.filter = blur;
    const on = pb < 0.5 ? 0 : 1;
    if (on !== bodyOn) { bodyItems[bodyOn].classList.remove('is-on'); bodyItems[on].classList.add('is-on'); bodyOn = on; }
    rail.style.transform = `scaleX(${pb})`;

    // nav sits on the render while a stage is pinned, gets a backing over regular content
    const pinned = hasGsap ? pins.some(p => p.isActive)
      : [hero, bodies].some(s => { const r = s.getBoundingClientRect(); return r.top <= 0 && r.bottom >= innerHeight; });
    nav.classList.toggle('solid', !pinned && scrollY > 8);
  }

  // ---------- live simulation, loaded on request ----------
  const live = document.querySelector('[data-sim]');
  live.querySelector('[data-sim-start]').addEventListener('click', async ev => {
    const btn = ev.currentTarget, note = live.querySelector('[data-sim-note]');
    btn.disabled = true; btn.textContent = 'Loading physics…';
    try {
      const { startSim } = await import('./sim.js');
      await startSim(live, { onProgress: p => { btn.textContent = `Loading physics… ${Math.round(p * 100)}%`; } });
      live.classList.add('is-live');
      live.querySelector('.live-ui').hidden = false;
    } catch (err) {
      console.error(err);
      btn.disabled = false; btn.textContent = 'Try again';
      note.textContent = 'The simulation could not start in this browser. Try a recent desktop browser; the renders above show the same robots.';
    }
  });

  function loop() {
    if (dirty) { dirty = false; render(); }
    requestAnimationFrame(loop);
  }
  addEventListener('scroll', () => (dirty = true), { passive: true });
  addEventListener('resize', () => (dirty = true));

  if (!reduced && window.Lenis) {
    const lenis = new Lenis({ lerp: 0.09, smoothWheel: true, anchors: true });
    lenis.on('scroll', () => (dirty = true));
    window.__lenis = lenis;
    if (hasGsap) {
      // one clock for both: Lenis advances on GSAP's ticker and ScrollTrigger reads every Lenis frame
      lenis.on('scroll', ScrollTrigger.update);
      gsap.ticker.add(t => lenis.raf(t * 1000));
      gsap.ticker.lagSmoothing(0);
    } else {
      const raf = t => { lenis.raf(t); requestAnimationFrame(raf); };
      requestAnimationFrame(raf);
    }
  }
  loop();
})();
