const chat = document.getElementById('chat');
const composer = document.getElementById('composer');
const input = document.getElementById('input');
const send = document.getElementById('send');

const history = [];

function addMessage(role, text) {
  const el = document.createElement('div');
  el.className = `msg ${role}`;
  el.textContent = text;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
  return el;
}

function addSources(el, toolsUsed) {
  if (!toolsUsed || toolsUsed.length === 0) return;
  const names = [...new Set(toolsUsed.map((t) => t.name.replaceAll('_', ' ')))];
  const sources = document.createElement('div');
  sources.className = 'sources';
  sources.textContent = `Sources queried: ${names.join(', ')}`;
  el.appendChild(sources);
}

composer.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  addMessage('user', text);
  history.push({ role: 'user', content: text });
  input.value = '';
  send.disabled = true;

  const pending = addMessage('assistant', 'Looking that up…');

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ history })
    });
    const data = await res.json();

    if (!res.ok) {
      pending.textContent = `Error: ${data.error ?? 'Request failed.'}`;
      pending.classList.add('system');
      return;
    }

    pending.textContent = data.reply;
    addSources(pending, data.toolsUsed);
    history.push({ role: 'assistant', content: data.reply });
  } catch (err) {
    pending.textContent = `Error: ${err.message}`;
  } finally {
    send.disabled = false;
    input.focus();
  }
});

composer.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    composer.requestSubmit();
  }
});
