(() => {
  const videos = [...document.querySelectorAll('.research-animation')];
  if (!videos.length || !('IntersectionObserver' in window)) return;
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const visible = new Set();
  // Respect a visitor pausing an animation with its native controls.
  const userPaused = new WeakSet();
  const updating = new WeakSet();

  function pause(video) {
    if (video.paused) return;
    updating.add(video);
    video.pause();
  }

  function play(video) {
    if (reduceMotion.matches || navigator.connection?.saveData ||
        document.hidden || userPaused.has(video)) return;
    video.play().catch(() => {}); // Controls remain available if autoplay is blocked.
  }

  videos.forEach(video => {
    const button = video.nextElementSibling;
    video.controls = false;
    button.hidden = false;
    const updateControl = () => {
      button.textContent = video.paused ? 'Play animation' : 'Pause animation';
    };
    button.addEventListener('click', () => {
      if (video.paused) video.play().catch(() => {});
      else video.pause();
    });
    // The muted attribute is also set as a property for Safari's autoplay policy.
    video.muted = true;
    video.addEventListener('pause', () => {
      if (updating.has(video)) updating.delete(video);
      else if (!video.ended) userPaused.add(video);
      updateControl();
    });
    video.addEventListener('play', () => {
      userPaused.delete(video);
      updateControl();
    });
  });

  const observer = new IntersectionObserver(entries => {
    entries.forEach(({ target: video, isIntersecting }) => {
      if (isIntersecting) {
        visible.add(video);
        play(video);
      } else {
        visible.delete(video);
        pause(video);
      }
    });
  }, { threshold: 0.1 });
  videos.forEach(video => observer.observe(video));
  document.addEventListener('visibilitychange', () => {
    videos.forEach(video => document.hidden ? pause(video) : visible.has(video) && play(video));
  });
  reduceMotion.addEventListener('change', () => {
    videos.forEach(video => reduceMotion.matches ? pause(video) : visible.has(video) && play(video));
  });
})();
