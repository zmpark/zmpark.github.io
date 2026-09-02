
(() => {
  const DPR = Math.min(window.devicePixelRatio || 1, 1.6);

  function init(section) {
    const canvas = document.createElement('canvas');
    canvas.className = 'particle-canvas';
    canvas.setAttribute('aria-hidden', 'true');
    section.prepend(canvas);

    const ctx = canvas.getContext('2d');
    let w = 0, h = 0, particles = [];
    const isHero = section.classList.contains('hero');
    const density = isHero ? 0.000095 : 0.00012;

    function resize() {
      const rect = section.getBoundingClientRect();
      w = Math.max(1, rect.width);
      h = Math.max(1, rect.height);
      canvas.width = Math.round(w * DPR);
      canvas.height = Math.round(h * DPR);
      canvas.style.width = w + 'px';
      canvas.style.height = h + 'px';
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);

      const count = Math.max(isHero ? 46 : 28, Math.round(w * h * density));
      particles = Array.from({length: count}, () => makeParticle(true));
    }

    function makeParticle(randomY = false) {
      const gold = Math.random() < 0.62;
      return {
        x: Math.random() * w,
        y: randomY ? Math.random() * h : h + Math.random() * 30,
        r: 0.8 + Math.random() * 1.65,
        vx: (Math.random() - 0.5) * 0.055,
        vy: -(0.025 + Math.random() * 0.075),
        phase: Math.random() * Math.PI * 2,
        twinkle: 0.65 + Math.random() * 1.2,
        gold
      };
    }

    function drawStar(x, y, r, alpha, gold) {
      ctx.save();
      ctx.translate(x, y);
      ctx.globalAlpha = alpha;
      ctx.fillStyle = gold ? 'rgba(237,211,145,1)' : 'rgba(255,255,255,1)';
      ctx.shadowColor = gold ? 'rgba(237,211,145,.72)' : 'rgba(255,255,255,.55)';
      ctx.shadowBlur = 8 + r * 4;
      ctx.beginPath();
      ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.fill();

      // Occasional tiny cross-glint
      if (r > 1.7) {
        ctx.shadowBlur = 4;
        ctx.globalAlpha = alpha * 0.55;
        ctx.fillRect(-r * 3.2, -0.35, r * 6.4, 0.7);
        ctx.fillRect(-0.35, -r * 3.2, 0.7, r * 6.4);
      }
      ctx.restore();
    }

    let t0 = performance.now();
    function animate(now) {
      const t = (now - t0) / 1000;
      ctx.clearRect(0, 0, w, h);

      // Very faint constellation links
      for (let i = 0; i < particles.length; i++) {
        const a = particles[i];
        for (let j = i + 1; j < particles.length; j++) {
          const b = particles[j];
          const dx = a.x - b.x, dy = a.y - b.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < 5200 && Math.random() < 0.018) {
            const d = Math.sqrt(d2);
            ctx.strokeStyle = `rgba(229,204,139,${0.035 * (1 - d / 72)})`;
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }

      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.y < -16 || p.x < -20 || p.x > w + 20) {
          Object.assign(p, makeParticle(false));
        }
        const alpha = 0.34 + 0.48 * (0.5 + 0.5 * Math.sin(t * p.twinkle + p.phase));
        drawStar(p.x, p.y, p.r, alpha, p.gold);
      }
      requestAnimationFrame(animate);
    }

    resize();
    new ResizeObserver(resize).observe(section);
    requestAnimationFrame(animate);
  }

  document.querySelectorAll('.hero, .page-banner').forEach(init);
})();
