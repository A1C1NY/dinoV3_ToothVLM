<script setup>
import { computed, nextTick, onMounted, ref } from 'vue'
import { ImagePlus, LoaderCircle, MessageCirclePlus, SendHorizontal, Trash2, X } from 'lucide-vue-next'
import { api } from './api'

const conversations = ref([])
const activeId = ref(null)
const messages = ref([])
const draft = ref('')
const attachments = ref([])
const loading = ref(false)
const error = ref('')
const health = ref({ ollama: 'checking', models: [] })
const messageList = ref(null)
const fileInput = ref(null)
let selectionRequest = 0

const activeConversation = computed(() => conversations.value.find((item) => item.id === activeId.value))

function formattedDate(value) {
  return new Intl.DateTimeFormat('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' }).format(new Date(value))
}

async function scrollToBottom() {
  await nextTick()
  window.scrollTo({ top: document.documentElement.scrollHeight, behavior: 'smooth' })
}

async function refreshConversations() {
  conversations.value = await api.conversations()
}

async function createConversation() {
  selectionRequest += 1
  error.value = ''
  const conversation = await api.createConversation()
  conversations.value.unshift(conversation)
  activeId.value = conversation.id
  messages.value = []
}

async function selectConversation(id) {
  const requestId = ++selectionRequest
  activeId.value = id
  error.value = ''
  const loadedMessages = await api.messages(id)
  if (requestId !== selectionRequest || activeId.value !== id) return
  const known = new Set()
  messages.value = loadedMessages.filter((message) => {
    if (known.has(message.id)) return false
    known.add(message.id)
    return true
  })
  await scrollToBottom()
}

async function removeConversation(id) {
  await api.deleteConversation(id)
  conversations.value = conversations.value.filter((item) => item.id !== id)
  if (activeId.value === id) {
    if (conversations.value[0]) await selectConversation(conversations.value[0].id)
    else await createConversation()
  }
}

function addFiles(event) {
  const selected = [...event.target.files].filter((file) => file.type.startsWith('image/'))
  attachments.value.push(...selected)
  event.target.value = ''
}

function removeAttachment(index) {
  attachments.value.splice(index, 1)
}

function messageText(event) {
  draft.value = event.target.value
}

async function submit() {
  if (loading.value || (!draft.value.trim() && !attachments.value.length) || !activeId.value) return
  loading.value = true
  error.value = ''
  const sentDraft = draft.value.trim()
  const sentFiles = [...attachments.value]
  draft.value = ''
  attachments.value = []
  const userMessageId = `local-user-${Date.now()}`
  const assistantMessageId = `local-assistant-${Date.now()}`
  messages.value.push({ id: userMessageId, conversation_id: activeId.value, role: 'user', content: sentDraft || '请分析我上传的口腔图片。', images: sentFiles.map((file) => URL.createObjectURL(file)), report: null })
  messages.value.push({ id: assistantMessageId, conversation_id: activeId.value, role: 'assistant', content: '', images: [], report: null, pending: true })
  await scrollToBottom()
  const conversationId = activeId.value
  try {
    const result = await api.sendMessage(conversationId, sentDraft, sentFiles)
    if (activeId.value !== conversationId) return
    const assistant = messages.value.find((message) => message.id === assistantMessageId)
    if (assistant) {
      assistant.pending = false
      assistant.report = result.user_message.report
      const text = result.assistant_message.content || ''
      assistant.content = ''
      for (const character of text) {
        assistant.content += character
        await new Promise((resolve) => setTimeout(resolve, 14))
      }
      Object.assign(assistant, result.assistant_message)
      assistant.content = text
    }
    const userIndex = messages.value.findIndex((message) => message.id === userMessageId)
    if (userIndex >= 0) messages.value[userIndex] = result.user_message
    await refreshConversations()
    await scrollToBottom()
  } catch (exception) {
    draft.value = sentDraft
    attachments.value = sentFiles
    messages.value = messages.value.filter((message) => message.id !== assistantMessageId)
    error.value = exception.message
  } finally {
    loading.value = false
  }
}

function onKeydown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    submit()
  }
}

onMounted(async () => {
  try {
    health.value = await api.health()
    await refreshConversations()
    if (conversations.value.length) await selectConversation(conversations.value[0].id)
    else await createConversation()
  } catch (exception) {
    error.value = exception.message
  }
})
</script>

<template>
  <main class="app-shell">
    <aside class="sidebar">
      <div class="brand"><span class="brand-mark">T</span><span>Tooth VLM</span></div>
      <button class="new-chat" type="button" @click="createConversation"><MessageCirclePlus :size="18" />新对话</button>
      <nav class="conversation-list" aria-label="对话列表">
        <button v-for="conversation in conversations" :key="conversation.id" type="button" class="conversation" :class="{ active: conversation.id === activeId }" @click="selectConversation(conversation.id)">
          <span class="conversation-title">{{ conversation.title }}</span>
          <span class="conversation-date">{{ formattedDate(conversation.updated_at) }}</span>
          <span class="delete-conversation" title="删除对话" @click.stop="removeConversation(conversation.id)"><Trash2 :size="15" /></span>
        </button>
      </nav>
      <div class="connection" :class="health.ollama">
        <span class="status-dot"></span>
        <span>{{ health.ollama === 'connected' ? `Ollama 已连接 · ${health.models.length} 个模型` : 'Ollama 未连接' }}</span>
      </div>
    </aside>

    <section class="chat-panel">
      <header class="chat-header">
        <div><h1>{{ activeConversation?.title || '新对话' }}</h1><p>口腔影像辅助分析</p></div>
        <div class="model-label">{{ health.selected_model || health.models?.[0] || '等待 Ollama' }}</div>
      </header>

      <div ref="messageList" class="messages">
        <div v-if="!messages.length" class="empty-state">
          <div class="empty-mark">T</div>
          <h2>开始一次口腔健康咨询</h2>
          <p>发送问题，或添加一张口腔图片进行辅助检测。</p>
        </div>
        <template v-for="message in messages" :key="message.id">
          <article class="message" :class="[message.role, { pending: message.pending }]">
            <div class="message-avatar">{{ message.role === 'assistant' ? 'T' : '我' }}</div>
            <div class="message-content">
              <div class="bubble"><template v-if="message.pending"><LoaderCircle :size="17" class="spin" />正在分析与生成回复</template><template v-else>{{ message.content }}</template></div>
              <div v-if="message.images?.length" class="image-grid">
                <img v-for="image in message.images" :key="image" :src="image" alt="用户上传的口腔图片" />
              </div>
            </div>
          </article>

          <article v-if="message.report?.analyses?.length" class="message assistant analysis-message">
            <div class="message-avatar">T</div>
            <div class="message-content">
              <p class="analysis-label">检测结果</p>
              <section class="analysis-results">
                <div v-for="(analysis, index) in message.report.analyses" :key="index">
                  <img v-if="analysis.annotated_image_url" :src="analysis.annotated_image_url" alt="检测标注结果" />
                  <p>{{ analysis.report || analysis.error }}</p>
                </div>
              </section>
            </div>
          </article>
        </template>
        <div v-if="loading" class="message assistant pending"><div class="message-avatar">T</div><div class="bubble"><LoaderCircle :size="17" class="spin" />正在分析与生成回复</div></div>
      </div>

      <footer class="composer-area">
        <p v-if="error" class="error-message">{{ error }}</p>
        <div v-if="attachments.length" class="attachments">
          <div v-for="(attachment, index) in attachments" :key="`${attachment.name}-${index}`" class="attachment"><span>{{ attachment.name }}</span><button type="button" title="移除图片" @click="removeAttachment(index)"><X :size="15" /></button></div>
        </div>
        <div class="composer">
          <input ref="fileInput" type="file" accept="image/jpeg,image/png,image/webp" multiple hidden @change="addFiles" />
          <button type="button" class="icon-button" title="添加图片" @click="fileInput.click()"><ImagePlus :size="21" /></button>
          <textarea :value="draft" rows="1" placeholder="输入问题，或添加口腔图片..." @input="messageText" @keydown="onKeydown"></textarea>
          <button type="button" class="send-button" title="发送" :disabled="loading || (!draft.trim() && !attachments.length)" @click="submit"><SendHorizontal :size="20" /></button>
        </div>
        <p class="disclaimer">本系统用于辅助筛查，不能替代专业牙科诊断。</p>
      </footer>
    </section>
  </main>
</template>
