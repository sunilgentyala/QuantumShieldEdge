/* ── Animated counters ─────────────────────────────────────────── */
function animateCounter(el, target, duration = 1400, suffix = '') {
  const start = performance.now();
  const isFloat = target % 1 !== 0;
  const step = ts => {
    const elapsed = Math.min((ts - start) / duration, 1);
    const ease = 1 - Math.pow(1 - elapsed, 3);
    const val = target * ease;
    el.textContent = (isFloat ? val.toFixed(1) : Math.floor(val)) + suffix;
    if (elapsed < 1) requestAnimationFrame(step);
    else el.textContent = (isFloat ? target.toFixed(1) : target) + suffix;
  };
  requestAnimationFrame(step);
}

/* ── Bar chart animation ───────────────────────────────────────── */
function animateBars() {
  document.querySelectorAll('.bar-fill[data-pct]').forEach(bar => {
    const pct = parseFloat(bar.dataset.pct);
    setTimeout(() => { bar.style.width = pct + '%'; }, 200);
  });
}

/* ── Intersection observer ─────────────────────────────────────── */
const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    const el = entry.target;

    if (el.classList.contains('metrics-strip')) {
      el.querySelectorAll('[data-count]').forEach(c => {
        const val = parseFloat(c.dataset.count);
        const sfx = c.dataset.suffix || '';
        animateCounter(c, val, 1200, sfx);
      });
    }
    if (el.classList.contains('results-section')) {
      animateBars();
    }
    observer.unobserve(el);
  });
}, { threshold: 0.2 });

document.querySelectorAll('.metrics-strip, .results-section').forEach(el => observer.observe(el));

/* ── Copy BibTeX ───────────────────────────────────────────────── */
document.querySelectorAll('.copy-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const pre = btn.closest('.citation-block').querySelector('pre');
    navigator.clipboard.writeText(pre.textContent.trim()).then(() => {
      btn.textContent = 'Copied!';
      btn.classList.add('copied');
      setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
    });
  });
});

/* ── Active nav link on scroll ─────────────────────────────────── */
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');
const onScroll = () => {
  let current = '';
  sections.forEach(s => {
    if (window.scrollY >= s.offsetTop - 100) current = s.id;
  });
  navLinks.forEach(a => {
    a.style.color = a.getAttribute('href') === '#' + current ? 'var(--text)' : '';
  });
};
window.addEventListener('scroll', onScroll, { passive: true });
