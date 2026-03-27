document.addEventListener('DOMContentLoaded', () => {
  // ===== Dark Mode Toggle =====
  const themeToggle = document.getElementById('theme-toggle');
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme === 'dark') {
    document.body.classList.add('dark-mode');
  }

  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      document.body.classList.toggle('dark-mode');
      const isDark = document.body.classList.contains('dark-mode');
      localStorage.setItem('theme', isDark ? 'dark' : 'light');
    });
  }

  // ===== Fullpage Section Scroll =====
  const isMobile = window.innerWidth <= 768;
  const snapSections = document.querySelectorAll('.snap-section');
  let currentIndex = 0;
  let isScrolling = false;
  let wheelTimeout = null;
  let wheelDelta = 0;
  const SCROLL_COOLDOWN = 1000;

  function scrollToSection(index) {
    if (index < 0 || index >= snapSections.length || isScrolling) return;
    isScrolling = true;
    currentIndex = index;
    snapSections[index].scrollIntoView({ behavior: 'smooth' });
    updateNavActive();
    triggerReveals(snapSections[index]);
    setTimeout(() => { isScrolling = false; }, SCROLL_COOLDOWN);
  }

  function updateNavActive() {
    const nav = document.querySelector('.nav-top');
    const navLinks = document.querySelectorAll('.nav-link');
    const currentId = snapSections[currentIndex].getAttribute('id');
    navLinks.forEach(link => {
      link.classList.toggle('active', link.getAttribute('href') === '#' + currentId);
    });
    nav.classList.toggle('scrolled', currentIndex > 0);
  }

  function triggerReveals(section) {
    section.querySelectorAll('.reveal, .reveal-left, .reveal-right').forEach(el => {
      el.classList.add('revealed');
    });
  }

  // Trigger reveals for the initial (hero) section
  triggerReveals(snapSections[0]);

  const timelineScroll = document.getElementById('timeline-scroll');
  const papersScrolls = document.querySelectorAll('.papers-scroll');

  if (!isMobile) {
    // Mouse wheel handler — batch rapid-fire events into one action
    // Allow internal scrolling on scrollable areas before jumping sections
    const scrollableAreas = [timelineScroll, ...papersScrolls];

    window.addEventListener('wheel', (e) => {
      for (const area of scrollableAreas) {
        if (area && area.matches(':hover') && area.scrollHeight > area.clientHeight) {
          const atTop = area.scrollTop <= 0;
          const atBottom = area.scrollTop + area.clientHeight >= area.scrollHeight - 1;
          if ((e.deltaY > 0 && !atBottom) || (e.deltaY < 0 && !atTop)) {
            return;
          }
        }
      }

      e.preventDefault();
      if (isScrolling) return;
      wheelDelta += e.deltaY;
      clearTimeout(wheelTimeout);
      wheelTimeout = setTimeout(() => {
        if (wheelDelta > 0) {
          scrollToSection(currentIndex + 1);
        } else if (wheelDelta < 0) {
          scrollToSection(currentIndex - 1);
        }
        wheelDelta = 0;
      }, 50);
    }, { passive: false });

    // Keyboard arrow keys & page up/down
    window.addEventListener('keydown', (e) => {
      if (isScrolling) return;
      if (e.key === 'ArrowDown' || e.key === 'PageDown' || e.key === ' ') {
        e.preventDefault();
        scrollToSection(currentIndex + 1);
      } else if (e.key === 'ArrowUp' || e.key === 'PageUp') {
        e.preventDefault();
        scrollToSection(currentIndex - 1);
      } else if (e.key === 'Home') {
        e.preventDefault();
        scrollToSection(0);
      } else if (e.key === 'End') {
        e.preventDefault();
        scrollToSection(snapSections.length - 1);
      }
    });
  }

  // On mobile, trigger reveals via IntersectionObserver instead of section jumps
  if (isMobile) {
    const revealElements = document.querySelectorAll('.reveal, .reveal-left, .reveal-right');
    const revealObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) entry.target.classList.add('revealed');
      });
    }, { threshold: 0.1, rootMargin: '0px 0px -30px 0px' });
    revealElements.forEach(el => revealObserver.observe(el));

    // Track active nav on mobile via scroll
    const mobileNavObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const id = entry.target.getAttribute('id');
          const nav = document.querySelector('.nav-top');
          document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.toggle('active', link.getAttribute('href') === '#' + id);
          });
          nav.classList.toggle('scrolled', id !== 'hero');
        }
      });
    }, { threshold: 0.3, rootMargin: '-50px 0px -40% 0px' });
    snapSections.forEach(s => mobileNavObserver.observe(s));
  }

  // Scroll-arrow clicks: find which section the arrow is in, go to next
  document.querySelectorAll('.scroll-arrow').forEach(arrow => {
    arrow.addEventListener('click', (e) => {
      e.preventDefault();
      if (isScrolling) return;
      const parentSection = arrow.closest('.snap-section');
      const idx = Array.from(snapSections).indexOf(parentSection);
      if (idx >= 0) scrollToSection(idx + 1);
    });
  });

  // Nav link clicks: jump to the correct section index
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      if (isScrolling) return;
      const targetId = link.getAttribute('href').slice(1);
      const idx = Array.from(snapSections).findIndex(s => s.id === targetId);
      if (idx >= 0) scrollToSection(idx);
    });
  });

  // Research subnav clicks
  document.querySelectorAll('.research-subnav a').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      if (isScrolling) return;
      const targetId = link.getAttribute('href').slice(1);
      const idx = Array.from(snapSections).findIndex(s => s.id === targetId);
      if (idx >= 0) scrollToSection(idx);
    });
  });

  // Intercept all internal anchor links (e.g. #genvision in timeline)
  document.addEventListener('click', (e) => {
    const link = e.target.closest('a[href^="#"]');
    if (!link) return;
    const targetId = link.getAttribute('href').slice(1);
    const idx = Array.from(snapSections).findIndex(s => s.id === targetId);
    if (idx >= 0) {
      e.preventDefault();
      if (!isScrolling) scrollToSection(idx);
    }
  });

  // ===== Papers scroll: hide hints when scrolled to bottom =====
  papersScrolls.forEach(ps => {
    const hint = ps.parentElement.querySelector('.papers-scroll-hint');
    // Hide hint if content doesn't overflow
    if (ps.scrollHeight <= ps.clientHeight && hint) {
      hint.classList.add('hidden');
    }
    if (hint) {
      ps.addEventListener('scroll', () => {
        const atBottom = ps.scrollTop + ps.clientHeight >= ps.scrollHeight - 5;
        hint.classList.toggle('hidden', atBottom);
      });
    }
  });

  // ===== Research Photo Slideshow (auto-discovers images/research/1.png, 2.png, ...) =====
  const researchContainer = document.getElementById('research-slideshow');
  if (researchContainer) {
    const researchSlides = [];
    let researchProbeIndex = 1;

    function probeResearchNext() {
      // Try .png first, then .jpg, then .JPG
      const extensions = ['png', 'jpg', 'JPG'];
      let tried = 0;

      function tryExt(extIdx) {
        if (extIdx >= extensions.length) {
          // No more extensions — done probing this index
          if (researchSlides.length > 0) {
            // Shuffle
            for (let i = researchSlides.length - 1; i > 0; i--) {
              const j = Math.floor(Math.random() * (i + 1));
              [researchSlides[i], researchSlides[j]] = [researchSlides[j], researchSlides[i]];
            }
            researchContainer.querySelectorAll('.research-slide').forEach(s => s.classList.remove('active'));
            researchSlides[0].classList.add('active');
            if (researchSlides.length > 1) {
              let cur = 0;
              setInterval(() => {
                researchSlides[cur].classList.remove('active');
                cur = (cur + 1) % researchSlides.length;
                researchSlides[cur].classList.add('active');
              }, 3000);
            }
          }
          return;
        }
        const img = new Image();
        const fname = researchProbeIndex + '.' + extensions[extIdx];
        img.src = 'images/research/' + fname;
        const idx = researchProbeIndex;
        img.onload = () => {
          img.classList.add('research-slide');
          img.alt = 'research photo ' + idx;
          if (fname === '2.png') img.classList.add('dim');
          if (researchSlides.length === 0) img.classList.add('active');
          researchContainer.appendChild(img);
          researchSlides.push(img);
          researchProbeIndex++;
          probeResearchNext();
        };
        img.onerror = () => {
          tryExt(extIdx + 1);
        };
      }

      tryExt(0);
    }

    probeResearchNext();
  }

  // ===== Timeline scroll hint =====
  const timelineHint = document.getElementById('timeline-hint');
  if (timelineScroll && timelineHint) {
    timelineScroll.addEventListener('scroll', () => {
      const atBottom = timelineScroll.scrollTop + timelineScroll.clientHeight >= timelineScroll.scrollHeight - 5;
      timelineHint.classList.toggle('hidden', atBottom);
    });
  }

  // ===== Hero Photo Slideshow (auto-discovers images/howard/1.JPG, 2.JPG, ...) =====
  const slideshowContainer = document.getElementById('hero-slideshow');
  if (slideshowContainer) {
    const loadedSlides = [];
    let probeIndex = 1;

    function probeNext() {
      const img = new Image();
      img.src = 'images/howard/' + probeIndex + '.JPG';
      const idx = probeIndex;
      img.onload = () => {
        img.classList.add('hero-slide');
        img.alt = 'photo ' + idx;
        if (loadedSlides.length === 0) img.classList.add('active');
        slideshowContainer.appendChild(img);
        loadedSlides.push(img);
        probeIndex++;
        probeNext();
      };
      img.onerror = () => {
        // No more images — shuffle and start the slideshow
        for (let i = loadedSlides.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [loadedSlides[i], loadedSlides[j]] = [loadedSlides[j], loadedSlides[i]];
        }
        // Reset active to first after shuffle
        slideshowContainer.querySelectorAll('.hero-slide').forEach(s => s.classList.remove('active'));
        if (loadedSlides.length > 0) loadedSlides[0].classList.add('active');
        if (loadedSlides.length > 1) startSlideshow();
      };
    }

    function startSlideshow() {
      let currentSlide = 0;
      setInterval(() => {
        loadedSlides[currentSlide].classList.remove('active');
        currentSlide = (currentSlide + 1) % loadedSlides.length;
        loadedSlides[currentSlide].classList.add('active');
      }, 3000);
    }

    probeNext();
  }

  // ===== Bio Photo Slideshow (same images as hero: images/howard/1.JPG, 2.JPG, ...) =====
  const bioContainer = document.getElementById('bio-slideshow');
  if (bioContainer) {
    const bioSlides = [];
    let bioProbeIndex = 1;

    function probeBioNext() {
      const img = new Image();
      img.src = 'images/howard/' + bioProbeIndex + '.JPG';
      const idx = bioProbeIndex;
      img.onload = () => {
        img.classList.add('bio-slide');
        img.alt = 'photo ' + idx;
        if (bioSlides.length === 0) img.classList.add('active');
        bioContainer.appendChild(img);
        bioSlides.push(img);
        bioProbeIndex++;
        probeBioNext();
      };
      img.onerror = () => {
        if (bioSlides.length > 1) {
          // Shuffle
          for (let i = bioSlides.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [bioSlides[i], bioSlides[j]] = [bioSlides[j], bioSlides[i]];
          }
          bioContainer.querySelectorAll('.bio-slide').forEach(s => s.classList.remove('active'));
          bioSlides[0].classList.add('active');
          let cur = 0;
          setInterval(() => {
            bioSlides[cur].classList.remove('active');
            cur = (cur + 1) % bioSlides.length;
            bioSlides[cur].classList.add('active');
          }, 3000);
        }
      };
    }

    probeBioNext();
  }

  // ===== Lyrics Auto-scroll =====
  let currentLine = 0;
  const lines = document.querySelectorAll('.lyrics-line');
  const lyricsContainer = document.getElementById('lyrics');

  if (lines.length > 0 && lyricsContainer) {
    lines[0].classList.add('current');

    function scrollLyrics() {
      lines[currentLine].classList.remove('current');
      currentLine = (currentLine + 1) % lines.length;
      lines[currentLine].classList.add('current');
      const scrollPos = -currentLine * 26;
      lyricsContainer.style.transform = `translateY(${scrollPos}px)`;
    }

    setInterval(scrollLyrics, 3000);
  }
});
