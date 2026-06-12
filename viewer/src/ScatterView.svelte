<script>
  import { onMount } from 'svelte'
  import * as PIXI from 'pixi.js'
  import { Viewport } from 'pixi-viewport'
  import { allPoints, visibleIds, showClusterOutlines } from './stores.js'

  let wrap
  let tip = { show: false, label: '', x: 0, y: 0 }

  const WORLD = 8000   // virtual canvas size along the shorter screen axis
  const THUMB = 56     // sprite size at world scale
  const OUTLINE = 2.5  // outline stroke width
  const PAD = 3        // gap between thumbnail edge and outline

  function worldDims(w, h) {
    // Scale world to match container aspect ratio so coords [0,1]² fill the screen
    return w >= h
      ? { ww: WORLD * (w / h), wh: WORLD }
      : { ww: WORLD, wh: WORLD * (h / w) }
  }

  onMount(() => {
    const app = new PIXI.Application({
      resizeTo: wrap,
      backgroundColor: 0x333333,
      autoDensity: true,
      resolution: window.devicePixelRatio || 1,
    })
    wrap.appendChild(app.view)

    let { ww, wh } = worldDims(wrap.clientWidth, wrap.clientHeight)

    const vp = new Viewport({
      screenWidth: wrap.clientWidth,
      screenHeight: wrap.clientHeight,
      worldWidth: ww,
      worldHeight: wh,
      events: app.renderer.events,
    })
    app.stage.addChild(vp)
    vp.drag().pinch().wheel().decelerate()
    vp.on('drag-start', () => { hasInteracted = true })
    vp.on('wheel', () => { hasInteracted = true })
    vp.on('pinch-start', () => { hasInteracted = true })

    const resetView = () => {
      hasInteracted = false
      vp.fit()
      vp.moveCenter(ww / 2, wh / 2)
    }
    app.view.addEventListener('dblclick', resetView)
    vp.sortableChildren = true

    const sprites = new Map()   // id -> Sprite
    const outlines = new Map()  // id -> Graphics
    const ptsById = new Map()   // id -> point data
    let unsubVisible = () => {}
    let unsubOutlines = () => {}
    let hasInteracted = false

    // Outline layer sits behind all sprites
    const outlineLayer = new PIXI.Container()
    vp.addChild(outlineLayer)

    const pts = $allPoints.filter(p => p.thumb)

    Promise.all(pts.map(p => PIXI.Texture.fromURL(p.thumb))).then(textures => {
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i]
        ptsById.set(p.id, p)

        // Outline (drawn behind sprite)
        if (p.color) {
          const half = THUMB / 2 + PAD
          const size = THUMB + PAD * 2
          const g = new PIXI.Graphics()
          g.lineStyle(OUTLINE, parseInt(p.color.replace('#', ''), 16), 0.9)
          g.drawRoundedRect(-half, -half, size, size, 4)
          g.x = p.x * ww
          g.y = p.y * wh
          outlineLayer.addChild(g)
          outlines.set(p.id, g)
        }

        // Sprite
        const s = new PIXI.Sprite(textures[i])
        s.width = THUMB
        s.height = THUMB
        s.anchor.set(0.5)
        s.x = p.x * ww
        s.y = p.y * wh
        s.eventMode = 'static'
        s.cursor = 'pointer'
        s.on('pointerenter', e => {
          tip = { show: true, label: p.path.split('/').pop(), x: e.global.x + 14, y: e.global.y - 6 }
          s.zIndex = 999
          s.width = THUMB * 1.4
          s.height = THUMB * 1.4
        })
        s.on('pointerleave', () => {
          tip = { ...tip, show: false }
          s.zIndex = 0
          s.width = THUMB
          s.height = THUMB
        })
        vp.addChild(s)
        sprites.set(p.id, s)
      }

      vp.fit()
      vp.moveCenter(ww / 2, wh / 2)

      unsubOutlines = showClusterOutlines.subscribe(show => {
        outlineLayer.visible = show
      })

      unsubVisible = visibleIds.subscribe(ids => {
        for (const [id, s] of sprites) s.visible = ids.has(id)
        for (const [id, g] of outlines) g.visible = ids.has(id)
      })
    })

    const ro = new ResizeObserver(() => {
      const dims = worldDims(wrap.clientWidth, wrap.clientHeight)
      ww = dims.ww; wh = dims.wh
      app.renderer.resize(wrap.clientWidth, wrap.clientHeight)
      vp.resize(wrap.clientWidth, wrap.clientHeight, ww, wh)
      for (const [id, s] of sprites) { const p = ptsById.get(id); s.x = p.x * ww; s.y = p.y * wh }
      for (const [id, g] of outlines) { const p = ptsById.get(id); g.x = p.x * ww; g.y = p.y * wh }
      if (!hasInteracted) {
        vp.fit()
        vp.moveCenter(ww / 2, wh / 2)
      }
    })
    ro.observe(wrap)

    return () => {
      unsubVisible()
      unsubOutlines()
      ro.disconnect()
      app.destroy(true, { children: true, texture: true })
    }
  })
</script>

<div class="wrap" bind:this={wrap}>
  {#if tip.show}
    <div class="tip" style="left:{tip.x}px;top:{tip.y}px">{tip.label}</div>
  {/if}
</div>

<style>
  .wrap { flex: 1; position: relative; overflow: hidden; }
  .tip {
    position: absolute; pointer-events: none; z-index: 10;
    background: rgba(0,0,0,.85); color: #eee;
    padding: 2px 8px; border-radius: 3px; font-size: 11px; white-space: nowrap;
  }
</style>
