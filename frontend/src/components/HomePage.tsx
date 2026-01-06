import { useEffect, useMemo, useState, useRef } from 'react'
import './HomePage.css'
import { ArrowRight, Clock, LayoutGrid, Search, Paperclip, X, Sun, Moon, Trash2 } from 'lucide-react'

type ThemeMode = 'dark' | 'light'

type CanvasSummary = {
  id: string
  name?: string
  createdAt?: number
  images?: unknown[]
  messages?: unknown[]
  data?: {
    elements?: any[]
    files?: Record<string, { dataURL?: string; mimeType?: string }>
  }
}

function getCanvasIdFromUrl() {
  try {
    const url = new URL(window.location.href)
    return url.searchParams.get('canvasId') || ''
  } catch {
    return ''
  }
}

function setCanvasIdToUrl(canvasId: string) {
  const url = new URL(window.location.href)
  url.searchParams.set('canvasId', canvasId)
  window.history.pushState({}, '', url.toString())
  window.dispatchEvent(new PopStateEvent('popstate'))
}

function clearCanvasIdFromUrl() {
  const url = new URL(window.location.href)
  url.searchParams.delete('canvasId')
  window.history.pushState({}, '', url.toString())
  window.dispatchEvent(new PopStateEvent('popstate'))
}

function pickThumbnails(canvas: CanvasSummary, max = 4): string[] {
  const files = canvas.data?.files || {}
  const urls = Object.values(files)
    .map((f) => (typeof f?.dataURL === 'string' ? f.dataURL : null))
    .filter((u): u is string => Boolean(u))
  // 去重 & 截断（最多展示 3 张拼贴预览）
  const uniq = Array.from(new Set(urls))
  return uniq.slice(0, max)
}

function countImages(canvas: CanvasSummary): number {
  const els = canvas.data?.elements || []
  const fromElements = els.filter((e) => e && !e.isDeleted && e.type === 'image').length
  if (fromElements) return fromElements
  return Array.isArray(canvas.images) ? canvas.images.length : 0
}

function extractTextPreview(canvas: CanvasSummary, maxLen = 84): string {
  const raw = Array.isArray(canvas.messages) ? canvas.messages : []
  // 从最新开始找一条“有可读文本”的消息
  for (let i = raw.length - 1; i >= 0; i--) {
    const m = raw[i] as any
    if (!m || typeof m !== 'object') continue
    const post = typeof m.postToolContent === 'string' ? m.postToolContent : ''
    const content = typeof m.content === 'string' ? m.content : ''
    let text = (post || content || '').trim()
    if (!text) continue

    // 清理掉图片标记/URL 列表，保留纯文本
    const lines = text
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean)
      .filter((l) => !l.startsWith('[图片:'))
      .filter((l) => l !== 'Generated Image:')
      .filter((l) => !(l.startsWith('- ') && (l.includes('/storage/') || l.includes('http://') || l.includes('https://'))))

    text = lines.join(' ').replace(/\s+/g, ' ').trim()
    if (!text) continue

    if (text.length > maxLen) return text.slice(0, maxLen) + '…'
    return text
  }
  return ''
}

export default function HomePage({
  theme,
  onToggleTheme,
}: {
  theme: ThemeMode
  onToggleTheme: () => void
}) {
  const [projects, setProjects] = useState<CanvasSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [query, setQuery] = useState('')
  const [prompt, setPrompt] = useState('')
  const [uploadedImages, setUploadedImages] = useState<string[]>([]) // 上传的图片URL列表
  const [creating, setCreating] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    // If already has canvasId (e.g. user pasted link), redirect to board
    const existing = getCanvasIdFromUrl()
    if (existing) return
    clearCanvasIdFromUrl()
  }, [])

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true)
        const res = await fetch('/api/canvases')
        if (!res.ok) throw new Error('Failed to load projects')
        const data = (await res.json()) as CanvasSummary[]
        setProjects(Array.isArray(data) ? data : [])
      } catch (e) {
        console.error(e)
        setProjects([])
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return projects
    return projects.filter((p) => (p.name || '').toLowerCase().includes(q) || p.id.toLowerCase().includes(q))
  }, [projects, query])

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

  const deleteProject = async (e: React.MouseEvent, projectId: string) => {
    e.stopPropagation() // 阻止点击卡片跳转
    if (!confirm('确定要删除这个项目吗？')) return
    
    try {
      await fetch(`/api/canvases/${projectId}`, { method: 'DELETE' })
      setProjects(prev => prev.filter(p => p.id !== projectId))
    } catch (e) {
      console.error('删除项目失败', e)
      alert('删除失败，请重试')
    }
  }

  const createProjectAndEnter = async (firstPrompt?: string, images?: string[]) => {
    if (creating) return
    setCreating(true)
    try {
      const id = `canvas-${Date.now()}`
      const nameFromPrompt = (firstPrompt || '').trim().slice(0, 18)
      const payload: CanvasSummary = {
        id,
        name: nameFromPrompt ? `项目：${nameFromPrompt}` : `项目 ${projects.length + 1}`,
        createdAt: Date.now(),
        images: [],
        data: { elements: [], files: {}, },
        messages: [],
      }
      await fetch('/api/canvases', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      // 构建消息内容：文本 + 图片URL
      let messageContent = (firstPrompt || '').trim()
      if (images && images.length > 0) {
        const imageTexts = images.map(url => `[图片: ${url}]`).join('\n')
        if (messageContent) {
          messageContent = `${messageContent}\n\n${imageTexts}`
        } else {
          messageContent = imageTexts
        }
      }

      if (messageContent) {
        // 存储完整的消息内容（包括图片）
        sessionStorage.setItem(`pending_prompt:${id}`, messageContent)
        // 同时存储图片URL列表，用于在对话界面显示
        if (images && images.length > 0) {
          sessionStorage.setItem(`pending_images:${id}`, JSON.stringify(images))
        }
      }
      setCanvasIdToUrl(id)
    } finally {
      setCreating(false)
    }
  }

  return (
    <div className="home">
      <div className="home__bg" />
      <header className="home__header">
        <div className="home__brand">
          <div className="home__logo">
            <img src="/logo.png" alt="PolyStudio" className="home__logo-img" />
          </div>
          <div>
            <div className="home__title">PolyStudio</div>
            <div className="home__subtitle">创意不止于画布</div>
          </div>
        </div>
        <div className="home__actions">
          <button className="home__btn home__btn--ghost" onClick={onToggleTheme} title="切换主题">
            {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
            {theme === 'dark' ? '亮色' : '暗色'}
          </button>
        </div>
      </header>

      <main className="home__main">
        <section className="home__hero">
          <h1 className="home__heroTitle">从问题开始，进入画板创作</h1>
          <p className="home__heroDesc">
            输入你的想法，我们会为你创建一个项目并自动进入画板。生成的图片会按顺序落到画布里。
          </p>

          <div className="home__promptCard">
            {/* 上传的图片预览 */}
            {uploadedImages.length > 0 && (
              <div className="home__uploaded-images">
                {uploadedImages.map((url, index) => (
                  <div key={index} className="home__uploaded-image-item">
                    <img src={url} alt={`上传的图片 ${index + 1}`} />
                    <button
                      className="home__remove-image-btn"
                      onClick={() => removeUploadedImage(index)}
                      title="移除图片"
                    >
                      <X size={14} />
                    </button>
                  </div>
                ))}
              </div>
            )}
            
            <div className="home__input-row">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                style={{ display: 'none' }}
              />
              <button
                className="home__upload-btn"
                onClick={() => fileInputRef.current?.click()}
                disabled={creating}
                title="上传图片"
              >
                <Paperclip size={18} />
              </button>
              <textarea
                className="home__promptInput"
                placeholder="例如：生成一套 12 生肖毛绒吉祥物，每个生肖一张图，统一风格…"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onPaste={(e) => {
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
                }}
                rows={3}
              />
            </div>
            <div className="home__promptFooter">
              <div className="home__hint">回车不会提交，点击右侧按钮创建项目并进入画板</div>
              <button
                className="home__btn home__btn--primary"
                onClick={() => createProjectAndEnter(prompt, uploadedImages)}
                disabled={creating || (!prompt.trim() && uploadedImages.length === 0)}
              >
                进入画板
                <ArrowRight size={16} />
              </button>
            </div>
          </div>
        </section>

        <section className="home__list">
          <div className="home__listHeader">
            <div className="home__listTitle">
              <LayoutGrid size={18} />
              历史项目
            </div>
            <div className="home__search">
              <Search size={16} />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="搜索项目名 / ID"
              />
            </div>
          </div>

          {loading ? (
            <div className="home__grid">
              {Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="home__card home__card--skeleton" />
              ))}
            </div>
          ) : (
            <div className="home__grid">
              {filtered.map((p) => {
                const thumbs = pickThumbnails(p)
                const created = p.createdAt ? new Date(p.createdAt).toLocaleString() : ''
                const n = countImages(p)
                const preview = extractTextPreview(p)
                return (
                  <div
                    key={p.id}
                    className="home__card-wrapper"
                  >
                    <button
                      className="home__card"
                      onClick={() => setCanvasIdToUrl(p.id)}
                      title={p.id}
                    >
                      <div className="home__thumb">
                        {thumbs.length > 0 ? (
                          <div
                            className={`home__thumbGrid home__thumbGrid--${Math.min(thumbs.length, 4)}`}
                            aria-hidden="true"
                          >
                            {thumbs.slice(0, 4).map((src, idx) => (
                              <img key={`${p.id}:${idx}`} className="home__thumbImg" src={src} alt="" />
                            ))}
                          </div>
                        ) : (
                          <div className="home__thumbEmpty">
                            {preview ? (
                              <>
                                <div className="home__thumbLabel">文本预览</div>
                                <div className="home__thumbText">{preview}</div>
                              </>
                            ) : (
                              'No Preview'
                            )}
                          </div>
                        )}
                      </div>
                      <div className="home__meta">
                        <div className="home__name">{p.name || '未命名项目'}</div>
                        {preview && <div className="home__preview">{preview}</div>}
                        <div className="home__sub">
                          <span className="home__chip">
                            <Clock size={14} />
                            {created || p.id}
                          </span>
                          <span className="home__chip">{n} 张</span>
                          <button
                            className="home__delete-chip"
                            onClick={(e) => deleteProject(e, p.id)}
                            title="删除项目"
                          >
                            <Trash2 size={14} />
                          </button>
                        </div>
                      </div>
                    </button>
                  </div>
                )
              })}
              {filtered.length === 0 && (
                <div className="home__empty">
                  没有匹配的项目。试试换个关键词，或者新建一个项目开始创作。
                </div>
              )}
            </div>
          )}
        </section>
      </main>
    </div>
  )
}


