async function request(url, options = {}) {
  const response = await fetch(url, options)
  if (response.status === 204) return null
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(body.detail || '请求失败，请稍后再试。')
  return body
}

export const api = {
  health: () => request('/api/health'),
  conversations: () => request('/api/conversations'),
  createConversation: () => request('/api/conversations', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}),
  }),
  deleteConversation: (id) => request(`/api/conversations/${id}`, { method: 'DELETE' }),
  messages: (id) => request(`/api/conversations/${id}/messages`),
  sendMessage: (id, prompt, images) => {
    const form = new FormData()
    form.append('prompt', prompt)
    images.forEach((image) => form.append('images', image))
    return request(`/api/conversations/${id}/messages`, { method: 'POST', body: form })
  },
}
