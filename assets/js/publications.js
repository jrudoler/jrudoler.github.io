(() => {
  const root = document.querySelector('.publications');
  if (!root) return;
  const list = root.querySelector('.publication-list');
  const papers = Array.from(list.children);
  const type = root.querySelector('#publication-type');
  const sort = root.querySelector('#publication-sort');
  let view = 'all';
  let topic = '';
  const topics = new Map(papers.map(paper => [paper, JSON.parse(paper.dataset.topics || 'null') || []]));

  function update() {
    const ordered = [...papers].sort((a, b) => {
      if (sort.value === 'title') return a.dataset.title.localeCompare(b.dataset.title);
      const direction = sort.value === 'oldest' ? 1 : -1;
      return direction * a.dataset.date.localeCompare(b.dataset.date) || a.id.localeCompare(b.id);
    });
    let count = 0;
    for (const paper of ordered) {
      paper.hidden = (view === 'selected' && paper.dataset.selected !== 'true') ||
        (topic && !topics.get(paper).includes(topic)) ||
        (type.value && paper.dataset.type !== type.value);
      if (!paper.hidden) count++;
      list.appendChild(paper);
    }
    root.querySelector('.publication-empty').hidden = count !== 0;
    root.querySelectorAll('[data-view]').forEach(button => button.setAttribute('aria-pressed', button.dataset.view === view));
    root.querySelectorAll('[data-topic]').forEach(button => button.setAttribute('aria-pressed', button.dataset.topic === topic));
  }
  root.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => {
    view = button.dataset.view; update();
  }));
  root.querySelectorAll('[data-topic]').forEach(button => button.addEventListener('click', () => {
    topic = button.dataset.topic; update();
  }));
  type.addEventListener('change', update);
  sort.addEventListener('change', update);
  root.querySelector('#publication-reset').addEventListener('click', () => {
    view = 'all'; topic = ''; type.value = ''; sort.value = 'newest'; update();
  });
  root.querySelector('.publication-controls').hidden = false;
  update();
})();
