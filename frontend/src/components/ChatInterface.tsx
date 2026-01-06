import { useState, useRef, useEffect, useMemo } from 'react'
import { Send, Paperclip, Image as ImageIcon, Sparkles, X, ChevronDown, ChevronRight, Link as LinkIcon, ArrowLeft, Sun, Moon } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import './ChatInterface.css'
import ExcalidrawCanvas, {
  ExcalidrawCanvasData,
  ExcalidrawCanvasHandle,
} from './ExcalidrawCanvas'

type ChatInterfaceProps = {
  initialCanvasId?: string
  theme: 'dark' | 'light'
  onToggleTheme: () => void
  onSetTheme: (t: 'dark' | 'light') => void
}

interface ToolCall {
  id: string
  name: string
  arguments: any
  status: 'executing' | 'done'
  result?: any
  imageUrl?: string
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  postToolContent?: string
  toolCalls?: ToolCall[]
  imageUrls?: string[] // 用户消息中的图片URL列表
}

interface CanvasImage {
  id: string
  url: string
  x: number
  y: number
  width: number
  height: number
}

interface Canvas {
  id: string
  name: string
  createdAt: number
  // Legacy: old DOM-drag canvas images
  images?: CanvasImage[]
  // New: Excalidraw canvas data
  data?: ExcalidrawCanvasData
  messages: Message[]
}

const ChatInterface = ({ initialCanvasId, theme, onToggleTheme, onSetTheme }: ChatInterfaceProps) => {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [uploadedImages, setUploadedImages] = useState<string[]>([]) // 上传的图片URL列表
  const fileInputRef = useRef<HTMLInputElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const chatMessagesRef = useRef<HTMLDivElement>(null)
  // 注意：为了实现“生成一次展示一次”的节奏，我们不再把所有工具调用塞进同一条 assistant 消息里。
  // delta 会写入最近的纯文本 assistant 消息；tool_call 会创建独立的 step 消息；tool_result 只更新对应 step。
  
  // 工具展开状态
  const [expandedTools, setExpandedTools] = useState<Set<string>>(new Set())

  // 画布管理状态
  const [canvases, setCanvases] = useState<Canvas[]>([])
  const [currentCanvasId, setCurrentCanvasId] = useState<string>('')

  const excalidrawRef = useRef<ExcalidrawCanvasHandle | null>(null)
  const [chatPanelCollapsed, setChatPanelCollapsed] = useState(false)
  const pendingSendRef = useRef<string | null>(null) // 标记待发送的消息

  const emptyCanvasData: ExcalidrawCanvasData = useMemo(
    () => ({ elements: [], appState: {}, files: {} }),
    []
  )

  const sanitizeCanvasData = (data: ExcalidrawCanvasData): ExcalidrawCanvasData => {
    const appState: any = data?.appState && typeof data.appState === 'object' ? { ...data.appState } : {}
    // Avoid Excalidraw crash after JSON persistence.
    if ('collaborators' in appState) {
      appState.collaborators = undefined
    }
    return {
      elements: Array.isArray(data?.elements) ? data.elements : [],
      files: (data?.files && typeof data.files === 'object') ? (data.files as any) : {},
      appState,
    }
  }

  const migrateLegacyCanvasToExcalidraw = (canvas: Canvas): Canvas => {
    if (canvas.data) {
      return { ...canvas, data: sanitizeCanvasData(canvas.data) }
    }
    const legacyImages = canvas.images || []
    if (legacyImages.length === 0) {
      return { ...canvas, data: emptyCanvasData }
    }

    const files: Record<string, any> = {}
    const elements: any[] = []

    for (const img of legacyImages) {
      const fileId = img.id || `im_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`
      files[fileId] = {
        id: fileId,
        dataURL: img.url,
        mimeType: 'image/png',
        created: Date.now(),
      }
      elements.push({
        type: 'image',
        id: fileId,
        x: img.x || 0,
        y: img.y || 0,
        width: img.width || 300,
        height: img.height || 300,
        angle: 0,
        fileId,
        strokeColor: '#000000',
        fillStyle: 'solid',
        strokeStyle: 'solid',
        boundElements: null,
        roundness: null,
        frameId: null,
        backgroundColor: 'transparent',
        strokeWidth: 1,
        roughness: 0,
        opacity: 100,
        groupIds: [],
        seed: Math.floor(Math.random() * 1_000_000),
        version: 1,
        versionNonce: Math.floor(Math.random() * 1_000_000),
        isDeleted: false,
        index: null,
        updated: Date.now(),
        link: null,
        locked: false,
        status: 'saved',
        scale: [1, 1],
        crop: null,
      })
    }

    return {
      ...canvas,
      data: sanitizeCanvasData({ elements, appState: {}, files }),
    }
  }


  // 初始化：加载画布列表
  useEffect(() => {
    fetchCanvases()
  }, [])

  const getCanvasLink = (canvasId: string) => {
    const url = new URL(window.location.href)
    url.searchParams.set('canvasId', canvasId)
    return url.toString()
  }

  const setCanvasIdInUrl = (canvasId: string) => {
    const url = new URL(window.location.href)
    url.searchParams.set('canvasId', canvasId)
    window.history.replaceState({}, '', url.toString())
  }

  const goHome = () => {
    const url = new URL(window.location.href)
    url.searchParams.delete('canvasId')
    window.history.pushState({}, '', url.toString())
    window.dispatchEvent(new PopStateEvent('popstate'))
  }

  const getCanvasIdFromUrl = () => {
    try {
      const url = new URL(window.location.href)
      return url.searchParams.get('canvasId') || ''
    } catch {
      return ''
    }
  }

  const fetchCanvases = async () => {
    try {
      const res = await fetch('/api/canvases')
      if (res.ok) {
        const data = await res.json()
        if (Array.isArray(data) && data.length > 0) {
          const migrated = data.map(migrateLegacyCanvasToExcalidraw)
          setCanvases(migrated)
          // 优先使用本地记录的选中ID，如果不存在则默认选中第一个
          const urlId = getCanvasIdFromUrl()
          const lastId = localStorage.getItem('ai_agent_current_canvas_id')
          const preferredId = initialCanvasId || urlId || lastId || ''
          const target = migrated.find((c: Canvas) => c.id === preferredId) || migrated[0]
          const canvasId = target.id
          setCurrentCanvasId(canvasId)
          setCanvasIdInUrl(canvasId)
          
          // 检查是否有待发送的消息（从首页来的）
          const pendingKey = `pending_prompt:${canvasId}`
          const pendingImagesKey = `pending_images:${canvasId}`
          const hasPending = sessionStorage.getItem(pendingKey)
          
          // 如果有待发送的消息，不设置后端消息，让 useEffect 处理
          if (hasPending) {
            setMessages([])
          } else {
            setMessages(target.messages || [])
          }
        } else {
          createNewCanvas()
        }
      } else {
        // 如果API失败，尝试创建新画布
        createNewCanvas()
      }
    } catch (e) {
      console.error('获取画布失败', e)
      createNewCanvas()
    }
  }

  const saveCanvasToBackend = async (canvas: Canvas) => {
    try {
      await fetch('/api/canvases', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(canvas)
      })
    } catch (e) {
      console.error('保存画布失败', e)
    }
  }

  // 当 messages 变化时，使用防抖机制同步更新到当前 canvas 并保存到后端
  useEffect(() => {
    if (!currentCanvasId) return
    
    // 防抖：只有当数据停止变化 5秒 后才执行保存，降低请求频率
    const timer = setTimeout(() => {
      setCanvases(prev => {
        const next = prev.map(canvas => {
          if (canvas.id === currentCanvasId) {
            // 检查是否有变更，避免不必要的请求
            if (JSON.stringify(canvas.messages) !== JSON.stringify(messages)) {
              const updatedCanvas = { ...canvas, messages }
              saveCanvasToBackend(updatedCanvas)
              return updatedCanvas
            }
          }
          return canvas
        })
        return next
      })
    }, 5000)

    return () => clearTimeout(timer)
  }, [messages, currentCanvasId])

  // 保存当前选中的画布ID到本地，方便刷新后恢复
  useEffect(() => {
    if (currentCanvasId) {
      localStorage.setItem('ai_agent_current_canvas_id', currentCanvasId)
    }
  }, [currentCanvasId])

  const createNewCanvas = async () => {
    const newCanvas: Canvas = {
      id: `canvas-${Date.now()}`,
      name: `项目 ${canvases.length + 1}`,
      createdAt: Date.now(),
      images: [],
      data: emptyCanvasData,
      messages: []
    }
    
    // 立即保存新画布
    await saveCanvasToBackend(newCanvas)
    
      setCanvases(prev => [newCanvas, ...prev])
      setCurrentCanvasId(newCanvas.id)
      setCanvasIdInUrl(newCanvas.id)
      setMessages([])
  }


  const copyCurrentCanvasLink = async () => {
    if (!currentCanvasId) return
    const link = getCanvasLink(currentCanvasId)
    try {
      await navigator.clipboard.writeText(link)
    } catch (e) {
      // fallback for some browsers / permissions
      window.prompt('复制这个链接：', link)
    }
  }


  const toggleToolDetails = (toolId: string) => {
    setExpandedTools(prev => {
      const next = new Set(prev)
      if (next.has(toolId)) {
        next.delete(toolId)
      } else {
        next.add(toolId)
      }
      return next
    })
  }

  const getCurrentCanvas = () => canvases.find((c) => c.id === currentCanvasId)

  const updateCurrentCanvasData = (updater: (prev: ExcalidrawCanvasData) => ExcalidrawCanvasData) => {
    setCanvases(prev => {
      const nextCanvases = prev.map(canvas => {
        if (canvas.id === currentCanvasId) {
          const base = migrateLegacyCanvasToExcalidraw(canvas)
          const newData = updater(base.data || emptyCanvasData)
          const updatedCanvas: Canvas = { ...base, data: newData }
          saveCanvasToBackend(updatedCanvas) 
          return updatedCanvas
        }
        return canvas
      })
      return nextCanvases
    })
  }

  const scrollToBottom = (behavior: ScrollBehavior = 'auto') => {
    const el = chatMessagesRef.current
    if (!el) return
    el.scrollTo({ top: el.scrollHeight, behavior })
  }

  useEffect(() => {
    scrollToBottom('auto')
  }, [messages])

  const sendMessage = async (userMessage: string, skipAddUserMessage = false) => {
    const trimmed = (userMessage || '').trim()
    if (!trimmed || isLoading) return
    setIsLoading(true)
    const newUserMessage: Message = {
      role: 'user',
      content: trimmed,
    }
    if (!skipAddUserMessage) {
      setMessages((prev) => [...prev, newUserMessage])
    }

    try {
      // 如果 skipAddUserMessage=true：用户消息通常已经被 setMessages 追加了，但 state 可能尚未刷新（闭包里还是旧 messages）
      // 为了保证后端一定能收到“用户刚发的这条”，这里做一次兜底合并。
      const messagesToUse = (() => {
        if (!skipAddUserMessage) return [...messages, newUserMessage]
        const last = messages[messages.length - 1]
        if (last && last.role === 'user' && (last.content || '').trim() === trimmed) return messages
        return [...messages, newUserMessage]
      })()
      const messageHistory = messagesToUse.map((msg) => {
        // 只把“可读文本 + 已生成图片URL”提供给模型作为上下文
        // （工具调用 UI 不进入历史；图片 URL 用于后续 edit_image 自动找到源图）
        let content = msg.content || ''
        if (msg.postToolContent) {
          content += '\n' + msg.postToolContent
        }
        if (msg.toolCalls) {
          const urls = msg.toolCalls
            .map((tc) => tc.imageUrl)
            .filter(Boolean) as string[]
          if (urls.length) {
            content += `\n\nGenerated Image:\n${urls.map((u) => `- ${u}`).join('\n')}`
        }
        }
        return { role: msg.role, content }
      })

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify({
          message: trimmed,
          messages: messageHistory.slice(0, -1),
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (!reader) throw new Error('无法读取响应流')

      let buffer = ''
      let eventCount = 0

      console.log('📡 开始接收流式数据...')

      const appendDelta = (deltaText: string) => {
        setMessages((prev) => {
          const next = [...prev]
          const last = next[next.length - 1]
          if (last && last.role === 'assistant' && (!last.toolCalls || last.toolCalls.length === 0)) {
            next[next.length - 1] = { ...last, content: (last.content || '') + deltaText }
            return next
          }
          next.push({ role: 'assistant', content: deltaText })
          return next
        })
      }

      const appendToolStep = (toolCall: ToolCall) => {
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: '',
            toolCalls: [toolCall],
          },
        ])
      }

      const updateToolStep = (toolCallId: string, updater: (tc: ToolCall) => ToolCall) => {
        setMessages((prev) => {
          const next = prev.map((m) => {
            if (!m.toolCalls) return m
            if (!m.toolCalls.some((tc) => tc.id === toolCallId)) return m
            return {
              ...m,
              toolCalls: m.toolCalls.map((tc) => (tc.id === toolCallId ? updater(tc) : tc)),
            }
          })
          return next
        })
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        buffer += chunk
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.trim() === '') continue
          
          if (line.startsWith('data: ')) {
            const data = line.slice(6).trim()
            if (data === '[DONE]') continue

            try {
              eventCount++
              const event = JSON.parse(data)

              switch (event.type) {
                case 'delta':
                  if (event.content) {
                    appendDelta(event.content)
                    setTimeout(() => scrollToBottom('auto'), 0)
                  }
                  break

                case 'tool_call':
                  appendToolStep({
                            id: event.id,
                            name: event.name,
                            arguments: event.arguments,
                    status: 'executing',
                  })
                  break
                
                case 'tool_call_chunk':
                  // 处理工具参数流
                  // 当前 UI 只展示 tool_call 的最终参数；
                  // 如需做“参数逐字流式展示”，可以在这里增量拼接。
                  break

                case 'tool_result':
                  updateToolStep(event.tool_call_id, (tc) => {
                           let updatedArgs = tc.arguments
                    let imageUrl: string | undefined = tc.imageUrl
                           try {
                             const resultObj = JSON.parse(event.content)
                             if (resultObj && resultObj.prompt && (!updatedArgs || Object.keys(updatedArgs).length === 0)) {
                               updatedArgs = { prompt: resultObj.prompt }
                             }
                      if (resultObj && typeof resultObj.image_url === 'string') {
                        imageUrl = resultObj.image_url
                             }
                           } catch (e) {
                             // ignore
                           }
                           return { 
                             ...tc, 
                             status: 'done' as const, 
                             result: event.content,
                      arguments: updatedArgs,
                      imageUrl,
                    }
                  })

                  if (event.content) {
                    try {
                      const result = JSON.parse(event.content)
                      if (typeof result.image_url === 'string' && result.image_url) {
                        const imgUrl: string = result.image_url
                        
                        // 新画布：将图片插入 Excalidraw（插入后会触发 onChange，从而自动保存到后端）
                        await excalidrawRef.current?.addImage({ url: imgUrl })
                        scrollToBottom('auto')
                      }
                    } catch (e) {
                      console.error('解析图片结果失败', e)
                    }
                  }
                  break

                case 'error':
                  setMessages((prev) => {
                    const newMessages = [...prev]
                    const lastMessage = newMessages[newMessages.length - 1]
                    if (lastMessage && lastMessage.role === 'assistant') {
                      lastMessage.content = `错误: ${event.error}`
                    }
                    return newMessages
                  })
                  break
              }
            } catch (e) {
              console.error('解析事件失败:', e)
            }
          }
        }
      }
    } catch (error) {
      console.error('请求失败:', error)
      setMessages((prev) => {
        const newMessages = [...prev]
        const lastMessage = newMessages[newMessages.length - 1]
        if (lastMessage && lastMessage.role === 'assistant') {
          lastMessage.content = `错误: ${error instanceof Error ? error.message : '未知错误'}`
        }
        return newMessages
      })
    } finally {
      setIsLoading(false)
      scrollToBottom('smooth')
    }
  }

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    
    // 验证文件类型
    if (!file.type.startsWith('image/')) {
      alert('只支持图片文件')
      return
    }
    
    try {
      const formData = new FormData()
      formData.append('file', file)
      
      const response = await fetch('/api/upload-image', {
        method: 'POST',
        body: formData,
      })
      
      if (!response.ok) {
        throw new Error('上传失败')
      }
      
      const data = await response.json()
      setUploadedImages(prev => [...prev, data.url])
    } catch (error) {
      console.error('图片上传失败:', error)
      alert('图片上传失败，请重试')
    } finally {
      // 清空文件选择，允许重复选择同一文件
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  const removeUploadedImage = (index: number) => {
    setUploadedImages(prev => prev.filter((_, i) => i !== index))
  }

  // 处理粘贴图片
  const handlePaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    try {
      const items = e.clipboardData?.items
      if (!items) return

      for (let i = 0; i < items.length; i++) {
        const item = items[i]
        if (item.type.indexOf('image') !== -1) {
          e.preventDefault()
          const file = item.getAsFile()
          if (!file) continue

          // 异步上传，但不阻塞
          (async () => {
            try {
              const formData = new FormData()
              formData.append('file', file)

              const response = await fetch('/api/upload-image', {
                method: 'POST',
                body: formData,
              })

              if (!response.ok) {
                throw new Error('上传失败')
              }

              const data = await response.json()
              setUploadedImages(prev => [...prev, data.url])
            } catch (error) {
              console.error('图片粘贴上传失败:', error)
              alert('图片粘贴上传失败，请重试')
            }
          })()
          break // 只处理第一张图片
        }
      }
    } catch (error) {
      console.error('粘贴处理错误:', error)
    }
  }

  const handleSend = async () => {
    if ((!input.trim() && uploadedImages.length === 0) || isLoading) return
    
    // 构建消息内容：文本 + 图片URL
    let messageContent = input.trim()
    const imageUrls = [...uploadedImages] // 保存图片URL列表
    
    if (uploadedImages.length > 0) {
      const imageTexts = uploadedImages.map(url => `[图片: ${url}]`).join('\n')
      if (messageContent) {
        messageContent = `${messageContent}\n\n${imageTexts}`
      } else {
        messageContent = imageTexts
      }
    }
    
    // 创建用户消息，包含图片URL列表
    const userMessageObj: Message = {
      role: 'user',
      content: messageContent,
      imageUrls: imageUrls.length > 0 ? imageUrls : undefined,
    }
    
    // 先添加到消息列表显示
    setMessages(prev => [...prev, userMessageObj])
    
    setInput('')
    setUploadedImages([]) // 清空上传的图片
    
    // 发送消息（skipAddUserMessage=true 因为已经添加了）
    await sendMessage(messageContent, true)
  }

  // 首页创建项目后，会把首条问题写入 sessionStorage：pending_prompt:<canvasId>
  // 进入画板时，先显示为用户消息，然后自动发送
  useEffect(() => {
    if (!currentCanvasId) return
    
    const key = `pending_prompt:${currentCanvasId}`
    const imagesKey = `pending_images:${currentCanvasId}`
    const pending = sessionStorage.getItem(key)
    const pendingImages = sessionStorage.getItem(imagesKey)
    
    if (!pending || !pending.trim()) return
    
    // 解析图片列表
    let imageUrls: string[] = []
    if (pendingImages) {
      try {
        imageUrls = JSON.parse(pendingImages) as string[]
      } catch (e) {
        console.error('解析图片列表失败', e)
      }
    }
    
    // 先显示为用户消息（显示在对话最前面）
    const userMessage: Message = {
      role: 'user',
      content: pending.trim(),
      imageUrls: imageUrls.length > 0 ? imageUrls : undefined,
    }
    
    // 清理 sessionStorage（在设置消息之前清理，避免重复处理）
    sessionStorage.removeItem(key)
    sessionStorage.removeItem(imagesKey)
    
    // 标记需要发送的消息
    pendingSendRef.current = pending.trim()
    
    // 设置消息（覆盖任何已有的消息）
    setMessages([userMessage])
  }, [currentCanvasId])
  
  // 监听消息变化，当消息设置完成后自动发送
  useEffect(() => {
    if (!pendingSendRef.current) return
    if (messages.length === 0) return
    if (isLoading) return
    
    const firstMessage = messages[0]
    // 检查消息内容是否匹配（支持纯文本或包含图片URL标记）
    const messageContent = firstMessage.content || ''
    const pendingContent = pendingSendRef.current
    
    // 匹配逻辑：直接匹配，或者消息内容包含pending内容（因为可能添加了图片标记）
    if (firstMessage.role === 'user' && 
        (messageContent === pendingContent || messageContent.includes(pendingContent))) {
      // 消息已设置，现在可以发送了
      const messageToSend = pendingSendRef.current
      pendingSendRef.current = null // 清除标记
      
      // 延迟发送，确保状态已更新，并且确保messages已经设置完成
      setTimeout(() => {
        // 确保messages中有这条消息，如果没有则重新添加
        setMessages(prev => {
          const hasMessage = prev.some(m => 
            m.role === 'user' && 
            (m.content === messageToSend || m.content?.includes(messageToSend))
          )
          if (!hasMessage) {
            return [...prev, { role: 'user', content: messageToSend, imageUrls: firstMessage.imageUrls }]
          }
          return prev
        })
        
        // 再延迟一点确保状态更新完成
        setTimeout(() => {
          sendMessage(messageToSend, true)
        }, 50)
      }, 150)
    }
  }, [messages, isLoading])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const cleanMessageContent = (content: string) => {
    if (!content) return ''
    // 不进行任何过滤，直接返回原始内容
    return content
  }

  // 获取工具显示名称
  const getToolDisplayName = (name: string) => {
    const map: Record<string, string> = {
      'generate_image': '生成图像',
      'edit_image': '编辑图像',
      'generate_volcano_image': '生成图像',
      'edit_volcano_image': '编辑图像',
    }
    return map[name] || name
  }

  // 获取当前画布的图片用于渲染
  const currentCanvas = getCurrentCanvas()
  const currentCanvasData =
    (currentCanvas ? migrateLegacyCanvasToExcalidraw(currentCanvas).data : emptyCanvasData) ||
    emptyCanvasData
  const hasAnyImages =
    (currentCanvasData?.elements || []).some((e: any) => e && !e.isDeleted && e.type === 'image')

  return (
    <div className="chat-interface">
      <div className="interface-layout">
        <div className="canvas-panel">
          {/* 画布控制栏 */}
          <div className="canvas-controls">
            <button className="control-btn" onClick={goHome} title="回到首页">
              <ArrowLeft size={18} />
              <span>首页</span>
            </button>
            <button className="control-btn" onClick={onToggleTheme} title="切换主题">
              {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
              <span>{theme === 'dark' ? '亮色' : '暗色'}</span>
            </button>

            <button
              className="control-btn"
              onClick={copyCurrentCanvasLink}
              title="复制项目链接"
              disabled={!currentCanvasId}
            >
              <LinkIcon size={18} />
              <span>复制链接</span>
            </button>
          </div>

          <div 
            className="canvas-content excalidraw-host-container"
          >
            {!hasAnyImages ? (
              <div className="canvas-empty">
                <ImageIcon size={64} strokeWidth={1.5} className="empty-icon" />
                <p className="empty-title">AI 画板</p>
                <p className="canvas-hint">生成的图片将自动落到画布上（支持缩放、框选、对齐）</p>
              </div>
            ) : (
              <div />
            )}

            {/* Excalidraw 画布（始终渲染，空态用 overlay 盖住即可） */}
            {currentCanvasId && (
              <ExcalidrawCanvas
                key={currentCanvasId}
                ref={excalidrawRef}
                canvasId={currentCanvasId}
                theme={theme}
                initialData={currentCanvasData}
                onDataChange={(data) => {
                  updateCurrentCanvasData(() => data)
                }}
                onThemeChange={(nextTheme) => {
                  if (nextTheme === 'dark' || nextTheme === 'light') {
                    onSetTheme(nextTheme)
                  }
                }}
                onImageToInput={async (url) => {
                  // 将图片添加到输入框
                  try {
                    // 如果是 data URL，需要先上传到服务器
                    if (url.startsWith('data:')) {
                      // 将 data URL 转换为 Blob 并上传
                      const response = await fetch(url)
                      const blob = await response.blob()
                      const formData = new FormData()
                      formData.append('file', blob, 'image.png')
                      
                      const uploadResponse = await fetch('/api/upload-image', {
                        method: 'POST',
                        body: formData,
                      })
                      
                      if (!uploadResponse.ok) {
                        throw new Error('上传失败')
                      }
                      
                      const data = await uploadResponse.json()
                      setUploadedImages(prev => [...prev, data.url])
                    } else if (url.startsWith('/storage/')) {
                      // 本地路径，直接使用
                      setUploadedImages(prev => [...prev, url])
                    } else {
                      // 其他 URL，可能需要处理
                      // 如果是 http/https，可能需要下载并上传
                      if (url.startsWith('http://') || url.startsWith('https://')) {
                        try {
                          const response = await fetch(url)
                          const blob = await response.blob()
                          const formData = new FormData()
                          formData.append('file', blob, 'image.png')
                          
                          const uploadResponse = await fetch('/api/upload-image', {
                            method: 'POST',
                            body: formData,
                          })
                          
                          if (uploadResponse.ok) {
                            const data = await uploadResponse.json()
                            setUploadedImages(prev => [...prev, data.url])
                          } else {
                            // 如果上传失败，尝试直接使用原 URL
                            setUploadedImages(prev => [...prev, url])
                          }
                        } catch (e) {
                          console.error('处理图片 URL 失败:', e)
                          // 失败时直接使用原 URL
                          setUploadedImages(prev => [...prev, url])
                        }
                      } else {
                        setUploadedImages(prev => [...prev, url])
                      }
                    }
                  } catch (err) {
                    console.error('处理图片失败:', err)
                    alert('添加图片到输入框失败，请重试')
                  }
                }}
                    />
            )}
          </div>
          
          {chatPanelCollapsed && (
            <button 
              className="floating-chat-btn"
              onClick={() => setChatPanelCollapsed(false)}
              title="展开对话"
            >
              <Sparkles size={24} />
            </button>
          )}
        </div>

        <div className={`chat-panel ${chatPanelCollapsed ? 'collapsed' : ''}`}>
          <div className="chat-header">
            <div className="header-title">
              <h1>PolyStudio</h1>
              <p>使用AI生成图像</p>
            </div>
            <button 
              className="close-chat-btn"
              onClick={() => setChatPanelCollapsed(true)}
              title="收起对话"
            >
              <X size={20} />
            </button>
          </div>

          <div className="chat-messages" ref={chatMessagesRef}>
            {messages.length === 0 && (
              <div className="empty-state">
                <div className="empty-icon-wrapper">
                  <Sparkles size={32} className="empty-icon-inner" />
                </div>
                <h3>开始创作</h3>
                <p>描述你想象中的画面，AI 帮你实现</p>
              </div>
            )}

            {messages.map((message, index) => (
              <div
                key={index}
                className={`message ${message.role === 'user' ? 'user-message' : 'assistant-message'}`}
              >
                <div className="message-content">
                  {message.role === 'assistant' ? (
                    <>
                      {/* 前置文本 */}
                      {message.content && (
                        <div className="message-text">
                          <ReactMarkdown>{cleanMessageContent(message.content)}</ReactMarkdown>
                        </div>
                      )}

                      {/* 工具调用 */}
                      {message.toolCalls && message.toolCalls.length > 0 && (
                        <div className="tool-calls-container">
                          {message.toolCalls.map((toolCall) => (
                            <div key={toolCall.id} className="tool-call-wrapper">
                              <div 
                                className={`tool-call-header ${toolCall.status === 'executing' ? 'executing' : 'done'}`}
                                onClick={() => toggleToolDetails(toolCall.id)}
                              >
                                <div className="tool-status-indicator">
                                  {toolCall.status === 'executing' ? (
                                    <div className="pulsing-dot" />
                                  ) : (
                                    <div className="status-dot done" />
                                  )}
                                </div>
                                <span className="tool-name">
                                  {getToolDisplayName(toolCall.name)}
                                </span>
                                <span className="tool-toggle-icon">
                                  {expandedTools.has(toolCall.id) ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                                </span>
                              </div>
                              
                              {expandedTools.has(toolCall.id) && (
                                <div className="tool-details">
                                  <div className="tool-section">
                                    <span className="section-label">输入参数</span>
                                    <pre>{JSON.stringify(toolCall.arguments, null, 2)}</pre>
                                  </div>
                                  {toolCall.result && (
                                    <div className="tool-section">
                                      <span className="section-label">执行结果</span>
                                      <pre>{
                                        typeof toolCall.result === 'string' 
                                          ? toolCall.result 
                                          : JSON.stringify(toolCall.result, null, 2)
                                      }</pre>
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}

                      {/* 图片显示 - 移到后置文本之前 */}
                      {/* 图片显示：按 toolCall 顺序逐个展示（每次工具调用一张图） */}
                      {message.toolCalls?.some(tc => tc.imageUrl) && (
                        <div className="message-images">
                          {message.toolCalls
                            .filter(tc => tc.imageUrl)
                            .map(tc => (
                              <div key={`img-${tc.id}`} className="message-image">
                                <img src={tc.imageUrl} alt="Generated" />
                              </div>
                            ))}
                        </div>
                      )}

                      {/* 后置消息文本内容 - 只有当有工具调用时才显示 */}
                      {message.postToolContent && (
                        <div className="message-text">
                          <ReactMarkdown>{cleanMessageContent(message.postToolContent)}</ReactMarkdown>
                        </div>
                      )}
                      
                      {/* 如果是最后一条消息且正在加载，显示光标 */}
                      {isLoading && index === messages.length - 1 && (
                        <span className="typing-cursor"></span>
                      )}
                    </>
                  ) : (
                    <>
                      {/* 用户消息中的图片 */}
                      {message.imageUrls && message.imageUrls.length > 0 && (
                        <div className="message-images">
                          {message.imageUrls.map((url, imgIndex) => (
                            <div key={`user-img-${imgIndex}`} className="message-image">
                              <img src={url} alt={`用户上传的图片 ${imgIndex + 1}`} />
                            </div>
                          ))}
                        </div>
                      )}
                      {/* 用户消息文本 */}
                      {(() => {
                        // 移除图片URL标记，只显示文本内容
                        const textContent = message.content
                          .split('\n')
                          .filter(line => !line.trim().startsWith('[图片:'))
                          .join('\n')
                          .trim()
                        return textContent ? (
                          <div className="message-text">{textContent}</div>
                        ) : message.imageUrls && message.imageUrls.length > 0 ? (
                          <div className="message-text" style={{ fontStyle: 'italic', color: '#9ca3af' }}>（已发送图片）</div>
                        ) : null
                      })()}
                    </>
                  )}
                </div>
              </div>
            ))}

            <div ref={messagesEndRef} />
          </div>

          <div className="chat-input-container">
            {/* 上传的图片预览 */}
            {uploadedImages.length > 0 && (
              <div className="uploaded-images-preview">
                {uploadedImages.map((url, index) => (
                  <div key={index} className="uploaded-image-item">
                    <img src={url} alt={`上传的图片 ${index + 1}`} />
                    <button
                      className="remove-image-btn"
                      onClick={() => removeUploadedImage(index)}
                      title="移除图片"
                    >
                      <X size={14} />
                    </button>
                  </div>
                ))}
              </div>
            )}
            
            <div className="input-row">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                style={{ display: 'none' }}
              />
              <button
                className="upload-image-button"
                onClick={() => fileInputRef.current?.click()}
                disabled={isLoading}
                title="上传图片"
              >
                <Paperclip size={18} />
              </button>
              <textarea
                className="chat-input"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                onPaste={handlePaste}
                placeholder="输入提示词生成图像..."
                rows={1}
                disabled={isLoading}
              />
              <button
                className="send-button"
                onClick={handleSend}
                disabled={isLoading || (!input.trim() && uploadedImages.length === 0)}
              >
                <Send size={18} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ChatInterface