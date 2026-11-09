(() => {
  const root = document.querySelector('.personal-interests');
  if (!root) return;
  const nav = root.querySelector('.interest-tabs');
  const tabs = [...nav.querySelectorAll('a')];
  const panels = tabs.map(tab => document.getElementById(tab.hash.slice(1)));
  nav.setAttribute('role', 'tablist');
  tabs.forEach((tab, index) => {
    tab.setAttribute('role', 'tab');
    tab.setAttribute('aria-controls', panels[index].id);
    panels[index].setAttribute('role', 'tabpanel');
    panels[index].tabIndex = 0;
  });

  function activate(index, updateUrl = false) {
    tabs.forEach((tab, i) => {
      const active = index === i;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
      panels[i].hidden = !active;
    });
    if (updateUrl) history.replaceState(null, '', tabs[index].hash);
  }

  // A deep link into a book category should open the Books panel too.
  function fromHash() {
    const target = document.getElementById(location.hash.slice(1));
    const index = panels.findIndex(panel => panel === target || panel.contains(target));
    activate(index < 0 ? 0 : index);
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', event => {
      event.preventDefault();
      activate(index, true);
    });
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next === undefined) return;
      event.preventDefault();
      activate(next, true);
      tabs[next].focus();
    });
  });
  window.addEventListener('hashchange', fromHash);
  fromHash();
})();
