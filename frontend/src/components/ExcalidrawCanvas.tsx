import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from 'react'

import { Excalidraw } from '@excalidraw/excalidraw'
import '@excalidraw/excalidraw/index.css'

type ExcalidrawFile = {
  id: string
  dataURL: string
  mimeType: string
  created: number
}

export type ExcalidrawCanvasData = {
  // Keep loose typing to avoid coupling to Excalidraw internals across versions.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  elements: readonly any[]
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  appState: any
  files: Record<string, ExcalidrawFile>
}

export type ExcalidrawCanvasHandle = {
  addImage: (args: { url: string }) => Promise<void>
  sendImageToInput: (callback: (url: string) => void) => void
}

type Props = {
  canvasId: string
  theme?: 'dark' | 'light'
  initialData?: ExcalidrawCanvasData
  onDataChange: (data: ExcalidrawCanvasData) => void
  onImageToInput?: (url: string) => void
  onThemeChange?: (theme: 'dark' | 'light') => void
}

function sanitizeAppState(appState: any) {
  if (!appState || typeof appState !== 'object') return {}

  // Excalidraw expects collaborators to be a Map. Once JSON-serialized, it often becomes `{}`,
  // which crashes inside Excalidraw (collaborators.forEach is not a function).
  // So we strip it on both load and save.
  const next = { ...appState }
  if ('collaborators' in next) {
    // Always remove to keep persisted data stable.
    // Excalidraw will re-initialize it internally.
    next.collaborators = undefined
  }
  return next
}

function sanitizeCanvasData(data: ExcalidrawCanvasData): ExcalidrawCanvasData {
  return {
    ...data,
    appState: sanitizeAppState(data.appState),
  }
}

function randomInt() {
  return Math.floor(Math.random() * 1_000_000)
}

function generateId(prefix: string) {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(16).slice(2, 8)}`
}

function guessMimeType(url: string) {
  const u = (url || '').toLowerCase()
  if (u.endsWith('.jpg') || u.endsWith('.jpeg')) return 'image/jpeg'
  if (u.endsWith('.webp')) return 'image/webp'
  if (u.endsWith('.gif')) return 'image/gif'
  return 'image/png'
}

function isMediaElement(el: any) {
  return (
    el &&
    !el.isDeleted &&
    (el.type === 'image' || el.type === 'embeddable' || el.type === 'video')
  )
}

function computeNextPosition(elements: any[], maxNumPerRow = 4, spacing = 20) {
  // 给左上角控制栏预留空间，避免第一张图与按钮挤在一起
  const baseX = 40
  const baseY = 120

  const media = (elements || []).filter(isMediaElement)
  if (media.length === 0) return { x: baseX, y: baseY }

  // Sort by top-left corner
  media.sort((a, b) => (a.y ?? 0) - (b.y ?? 0) || (a.x ?? 0) - (b.x ?? 0))

  // Group into rows by vertical overlap
  const rows: any[][] = []
  for (const el of media) {
    const y = el.y ?? 0
    const h = el.height ?? 0
    let placed = false
    for (const row of rows) {
      const overlaps = row.some((r) => {
        const ry = r.y ?? 0
        const rh = r.height ?? 0
        return Math.max(y, ry) < Math.min(y + h, ry + rh)
      })
      if (overlaps) {
        row.push(el)
        placed = true
        break
      }
    }
    if (!placed) rows.push([el])
  }

  rows.sort((ra, rb) => {
    const ay = ra.reduce((s, e) => s + (e.y ?? 0), 0) / ra.length
    const by = rb.reduce((s, e) => s + (e.y ?? 0), 0) / rb.length
    return ay - by
  })

  const lastRow = rows[rows.length - 1]
  lastRow.sort((a, b) => (a.x ?? 0) - (b.x ?? 0))

  if (lastRow.length < maxNumPerRow) {
    const right = lastRow[lastRow.length - 1]
    return {
      x: Math.max(baseX, (right.x ?? 0) + (right.width ?? 0) + spacing),
      y: Math.max(baseY, Math.min(...lastRow.map((e) => e.y ?? 0))),
    }
  }

  const bottom = Math.max(...lastRow.map((e) => (e.y ?? 0) + (e.height ?? 0)))
  return { x: baseX, y: Math.max(baseY, bottom + spacing) }
}

export const ExcalidrawCanvas = forwardRef<ExcalidrawCanvasHandle, Props>(
  ({ canvasId, theme, initialData, onDataChange, onImageToInput, onThemeChange }, ref) => {
    const [api, setApi] = useState<any>(null)
    const saveTimer = useRef<number | null>(null)
    const imageToInputCallbackRef = useRef<((url: string) => void) | null>(null)
    const lastThemeRef = useRef<'dark' | 'light' | null>(null)
    const lastSaveTimeRef = useRef<number>(0) // 记录上次保存时间
    const periodicSaveTimer = useRef<number | null>(null) // 30秒定时保存

    const flushSave = useCallback(
      (data: ExcalidrawCanvasData) => {
        if (saveTimer.current) {
          window.clearTimeout(saveTimer.current)
          saveTimer.current = null
        }
        onDataChange(sanitizeCanvasData(data))
      },
      [onDataChange]
    )

    const snapshotScene = useCallback((): ExcalidrawCanvasData | null => {
      try {
        if (!api) return null
        const elements = api.getSceneElements ? api.getSceneElements() : null
        const appState = api.getAppState ? api.getAppState() : null
        const files = api.getFiles ? api.getFiles() : null
        if (!elements || !files) return null
        return sanitizeCanvasData({
          elements,
          appState: appState || {},
          files,
        })
      } catch {
        return null
      }
    }, [api])

    // 画布变化时立即保存（短防抖，避免频繁操作时请求过多）
    const debouncedSave = useCallback(
      (data: ExcalidrawCanvasData) => {
        if (saveTimer.current) {
          window.clearTimeout(saveTimer.current)
        }
        // 画布变化时，短防抖500ms后保存
        saveTimer.current = window.setTimeout(() => {
          onDataChange(sanitizeCanvasData(data))
          // 记录保存时间，用于30秒定时保存判断
          lastSaveTimeRef.current = Date.now()
        }, 500)
      },
      [onDataChange]
    )

    useEffect(() => {
      // 30秒定时保存（仅在距离上次保存超过30秒时才保存）
      const startPeriodicSave = () => {
        if (periodicSaveTimer.current) {
          window.clearTimeout(periodicSaveTimer.current)
        }
        periodicSaveTimer.current = window.setTimeout(() => {
          const now = Date.now()
          // 如果距离上次保存超过30秒，才保存
          if (now - lastSaveTimeRef.current >= 30000) {
            const snap = snapshotScene()
            if (snap) {
              flushSave(snap)
              lastSaveTimeRef.current = now
            }
          }
          // 继续下一轮定时
          startPeriodicSave()
        }, 30000) // 每30秒检查一次
      }

      // 页面隐藏/切换标签时：如果距离上次保存超过30秒，才保存
      const onVisibilityChange = () => {
        if (document.hidden) {
          const now = Date.now()
          if (now - lastSaveTimeRef.current >= 30000) {
            const snap = snapshotScene()
            if (snap) {
              flushSave(snap)
              lastSaveTimeRef.current = now
            }
          }
        }
      }

      // 刷新/关闭/跳转时：如果距离上次保存超过30秒，才保存
      const onBeforeUnload = () => {
        // 清除防抖定时器
        if (saveTimer.current) {
          window.clearTimeout(saveTimer.current)
          saveTimer.current = null
        }
        const now = Date.now()
        // 如果距离上次保存超过30秒，才保存
        if (now - lastSaveTimeRef.current >= 30000) {
          const snap = snapshotScene()
          if (snap) {
            flushSave(snap)
            lastSaveTimeRef.current = now
          }
        }
      }

      // 启动30秒定时保存
      startPeriodicSave()

      document.addEventListener('visibilitychange', onVisibilityChange)
      window.addEventListener('beforeunload', onBeforeUnload)

      return () => {
        document.removeEventListener('visibilitychange', onVisibilityChange)
        window.removeEventListener('beforeunload', onBeforeUnload)
        if (periodicSaveTimer.current) {
          window.clearTimeout(periodicSaveTimer.current)
          periodicSaveTimer.current = null
        }
        if (saveTimer.current) {
          window.clearTimeout(saveTimer.current)
          saveTimer.current = null
        }
      }
    }, [flushSave, snapshotScene])

    // 外部主题 -> Excalidraw（关键：让“切换主题”对画板生效）
    useEffect(() => {
      if (!api) return
      if (theme !== 'dark' && theme !== 'light') return
      if (lastThemeRef.current === theme) return
      lastThemeRef.current = theme
      try {
        // updateScene 会合并 appState；这里仅更新 theme，避免污染其它状态
        api.updateScene({ appState: { theme } })
      } catch (e) {
        // ignore
      }
    }, [api, theme])

    const initial = useMemo(() => {
      // Excalidraw expects initialData to be plain object; we keep it as-is.
      if (!initialData) return null
      return sanitizeCanvasData(initialData)
    }, [initialData])

    // 拼接多张图片为一张
    const combineImages = async (imageUrls: string[]): Promise<string> => {
      return new Promise((resolve, reject) => {
        const images: HTMLImageElement[] = []
        let loadedCount = 0

        const loadImage = (url: string, index: number) => {
          const img = new Image()
          img.crossOrigin = 'anonymous'
          img.onload = () => {
            images[index] = img
            loadedCount++
            if (loadedCount === imageUrls.length) {
              // 所有图片加载完成，开始拼接
              const maxWidth = Math.max(...images.map(img => img.width))
              const maxHeight = Math.max(...images.map(img => img.height))
              
              // 计算网格布局：尽量接近正方形
              const count = images.length
              const cols = Math.ceil(Math.sqrt(count))
              const rows = Math.ceil(count / cols)
              
              const canvas = document.createElement('canvas')
              canvas.width = maxWidth * cols
              canvas.height = maxHeight * rows
              const ctx = canvas.getContext('2d')
              
              if (!ctx) {
                reject(new Error('无法创建 canvas context'))
                return
              }
              
              // 填充白色背景
              ctx.fillStyle = '#ffffff'
              ctx.fillRect(0, 0, canvas.width, canvas.height)
              
              // 绘制每张图片
              images.forEach((img, index) => {
                const col = index % cols
                const row = Math.floor(index / cols)
                const x = col * maxWidth
                const y = row * maxHeight
                
                // 居中绘制
                const offsetX = (maxWidth - img.width) / 2
                const offsetY = (maxHeight - img.height) / 2
                
                ctx.drawImage(img, x + offsetX, y + offsetY)
              })
              
              // 转换为 data URL
              resolve(canvas.toDataURL('image/png'))
            }
          }
          img.onerror = () => {
            reject(new Error(`加载图片失败: ${url}`))
          }
          img.src = url
        }

        imageUrls.forEach((url, index) => {
          loadImage(url, index)
        })
      })
    }

    // 监听右键菜单，添加"发送到输入框"选项
    useEffect(() => {
      if (!api || !onImageToInput) return

      let processTimer: number | null = null

      // 处理菜单的函数
      const processMenu = (contextMenu: Element) => {
        // 检查是否已经添加过菜单项（通过检查 DOM 中是否已存在）
        const existingItem = contextMenu.querySelector('[data-action="send-to-input"]')
        if (existingItem) {
          return // 已经添加过了，不再重复添加
        }
        
        // 给菜单容器打上我们自己的 class，方便精准美化
        try {
          ;(contextMenu as HTMLElement).classList.add('polystudio-excalidraw-menu')
        } catch {
          // ignore
        }
        
        // 创建"发送到输入框"菜单项（始终显示，像其他菜单项一样）
        const menuItem = document.createElement('div')
        menuItem.className = 'context-menu-item polystudio-context-menu-item'
        menuItem.setAttribute('data-action', 'send-to-input')
        menuItem.textContent = '发送到输入框'
        
        menuItem.addEventListener('click', async (e) => {
          e.preventDefault()
          e.stopPropagation()
          
          // 点击时检查是否有选中的图片
          const currentElements = api.getSceneElements() || []
          const currentAppState = api.getAppState()
          const currentSelectedIds = currentAppState?.selectedElementIds || {}
          
          const currentSelectedImages = currentElements.filter((el: any) => {
            if (!el || el.isDeleted || el.type !== 'image') return false
            return currentSelectedIds[el.id] || false
          })
          
          if (currentSelectedImages.length === 0) {
            // 没有选中图片，提示用户
            alert('请先选中图片')
            return
          }
          
          const files = api.getFiles?.() || {}
          const imageUrls = currentSelectedImages
            .map((el: any) => {
              const file = files[el.fileId]
              return file?.dataURL
            })
            .filter(Boolean) as string[]
          
          if (imageUrls.length === 0) {
            alert('无法获取图片数据')
            return
          }
          
          try {
            if (imageUrls.length === 1) {
              // 单张图片，直接发送
              onImageToInput(imageUrls[0])
            } else {
              // 多张图片，拼接后发送
              const combinedImage = await combineImages(imageUrls)
              onImageToInput(combinedImage)
            }
          } catch (error) {
            console.error('处理图片失败:', error)
            alert('处理图片失败，请重试')
          }
          
          // 关闭菜单
          const menu = contextMenu.closest('.context-menu') || contextMenu
          if (menu && menu.parentNode) {
            menu.parentNode.removeChild(menu)
          }
        })

        // 插入到菜单的最前面（第一个位置）
        const firstChild = contextMenu.firstElementChild
        if (firstChild) {
          contextMenu.insertBefore(menuItem, firstChild)
        } else {
          contextMenu.appendChild(menuItem)
        }
        
        // 隐藏所有分隔线（包括 Excalidraw 自己的）
        const allSeparators = contextMenu.querySelectorAll('hr, [role="separator"], .polystudio-context-separator')
        allSeparators.forEach((sep: Element) => {
          ;(sep as HTMLElement).style.display = 'none'
        })
        
        // 隐藏一些不常用的菜单项，精简菜单
        const menuItems = contextMenu.querySelectorAll('[role="menuitem"], .context-menu-item, button')
        menuItems.forEach((item: Element) => {
          const text = item.textContent || ''
          // 隐藏一些不常用的选项
          const hideItems = [
            'Wrap selection in frame',
            '复制为PNG 到剪贴板',
            '复制为 SVG 到剪贴板',
            '拷贝样式',
            '粘贴样式',
            '添加到素材库中',
            '全部锁定',
            'Copy link',
            'Copy link to object',
            '复制链接',
            '复制链接到对象',
          ]
          if (hideItems.some(hideText => text.includes(hideText))) {
            ;(item as HTMLElement).style.display = 'none'
          }
        })
      }

      // 使用 MutationObserver 监听右键菜单的出现
      const observer = new MutationObserver(() => {
        // 查找 Excalidraw 的右键菜单（可能有多种选择器）
        const contextMenu = document.querySelector('.context-menu') || 
                           document.querySelector('[class*="context-menu"]') ||
                           document.querySelector('[role="menu"]') ||
                           document.querySelector('.dropdown-menu')

        if (!contextMenu) {
          if (processTimer) {
            clearTimeout(processTimer)
            processTimer = null
          }
          return
        }
        
        // 清除之前的定时器
        if (processTimer) {
          clearTimeout(processTimer)
        }
        
        // 延迟处理，确保菜单完全渲染后再添加菜单项
        processTimer = window.setTimeout(() => {
          processMenu(contextMenu)
          processTimer = null
        }, 50)
      })

      // 观察整个文档的变化
      observer.observe(document.body, {
        childList: true,
        subtree: true,
      })

      return () => {
        observer.disconnect()
        if (processTimer) {
          clearTimeout(processTimer)
        }
      }
    }, [api, onImageToInput, combineImages])

    useImperativeHandle(
      ref,
      () => ({
        sendImageToInput: (callback: (url: string) => void) => {
          imageToInputCallbackRef.current = callback
          // 获取当前选中的图片
          if (!api) return
          const elements = api.getSceneElements() || []
          const appState = api.getAppState()
          const selectedElementIds = appState?.selectedElementIds || {}
          
          const selectedImages = elements.filter((el: any) => {
            if (!el || el.isDeleted || el.type !== 'image') return false
            return selectedElementIds[el.id] || false
          })

          if (selectedImages.length > 0) {
            const imageEl = selectedImages[0]
            const files = api.getFiles?.() || {}
            const file = files[imageEl.fileId]
            if (file && file.dataURL) {
              callback(file.dataURL)
            }
          }
        },
        addImage: async ({ url }) => {
          if (!api) return

          const img = new Image()
          img.crossOrigin = 'anonymous'

          const { width, height } = await new Promise<{ width: number; height: number }>(
            (resolve) => {
              img.onload = () => resolve({ width: img.naturalWidth || 1024, height: img.naturalHeight || 1024 })
              img.onerror = () => resolve({ width: 1024, height: 1024 })
              img.src = url
            }
          )

          const maxW = 300
          const scale = width > 0 ? Math.min(1, maxW / width) : 1
          const finalW = Math.max(32, Math.round(width * scale))
          const finalH = Math.max(32, Math.round(height * scale))

          const fileId = generateId('im')
          const created = Date.now()

          const file: ExcalidrawFile = {
            id: fileId,
            dataURL: url,
            mimeType: guessMimeType(url),
            created,
          }

          const elements = api.getSceneElements() || []
          const { x, y } = computeNextPosition(elements)

          // 在图片背后加一层白底板，避免深色画布背景导致“同图不同观感”（聊天里通常像白底卡片）
          const bgId = generateId('bg')
          const bgElement = {
            type: 'rectangle',
            id: bgId,
            x,
            y,
            width: finalW,
            height: finalH,
            angle: 0,
            strokeColor: 'transparent',
            backgroundColor: '#ffffff',
            fillStyle: 'solid',
            strokeStyle: 'solid',
            strokeWidth: 1,
            roughness: 0,
            opacity: 100,
            groupIds: [],
            seed: randomInt(),
            version: 1,
            versionNonce: randomInt(),
            isDeleted: false,
            boundElements: null,
            roundness: null,
            frameId: null,
            updated: created,
            link: null,
            locked: false,
          }

          const newElement = {
            type: 'image',
            id: fileId,
            x,
            y,
            width: finalW,
            height: finalH,
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
            seed: randomInt(),
            version: 1,
            versionNonce: randomInt(),
            isDeleted: false,
            index: null,
            updated: created,
            link: null,
            locked: false,
            status: 'saved',
            scale: [1, 1],
            crop: null,
          }

          api.addFiles([file])
          // 顺序很重要：先底板再图片，保证底板在下方
          const nextElements = [...elements, bgElement, newElement]
          api.updateScene({ elements: nextElements })

          // 关键：插入图片后立即强制保存一次（画布变化，立即保存）
          const nextFiles = { ...(api.getFiles?.() || {}), [fileId]: file }
          const nextAppState = api.getAppState ? api.getAppState() : {}
          flushSave({
            elements: nextElements,
            appState: nextAppState,
            files: nextFiles,
          })
          // 记录保存时间
          lastSaveTimeRef.current = Date.now()
        },
      }),
      [api, flushSave]
    )

    return (
      <div className="excalidraw-host" data-canvas-id={canvasId}>
        <Excalidraw
          langCode="zh-CN"
          excalidrawAPI={(instance) => setApi(instance)}
          initialData={initial as any}
          onChange={(elements: readonly any[], appState: any, files: any) => {
            const nextTheme = appState?.theme === 'light' ? 'light' : appState?.theme === 'dark' ? 'dark' : null
            if (nextTheme && nextTheme !== lastThemeRef.current) {
              lastThemeRef.current = nextTheme
              onThemeChange?.(nextTheme)
            }
            const data: ExcalidrawCanvasData = sanitizeCanvasData({
              elements,
              appState: appState,
              files,
            })
            debouncedSave(data)
          }}
        />
      </div>
    )
  }
)

ExcalidrawCanvas.displayName = 'ExcalidrawCanvas'

export default ExcalidrawCanvas


