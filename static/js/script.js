document.addEventListener('DOMContentLoaded', () => {
    const title = document.getElementById('animated-title');
    if (title) {
      const text = title.textContent;
      title.innerHTML = text.split('').map((char, i) => {
        const space = char === ' ' ? '&nbsp;' : char;
        return `<span class="letter" style="--delay:${i * 0.05}s">${space}</span>`;
      }).join('');
    }
  });