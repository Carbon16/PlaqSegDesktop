<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { listen, type UnlistenFn } from '@tauri-apps/api/event';
  import { commands } from '$lib/invoke';
  import type {
    ModelInfo,
    InferenceConfig,
    ImageResult,
    QueuedImage,
    ProgressEvent,
  } from '$lib/types';
  import { DEFAULT_CONFIG } from '$lib/types';

  // ── Theme ────────────────────────────────────────────────────────────────────
  let theme = $state<'dark' | 'light'>('dark');
  $effect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('plaqseg-theme', theme);
  });

  // ── State ────────────────────────────────────────────────────────────────────
  let modelsFolder = $state('');
  let availableModels = $state<ModelInfo[]>([]);
  let selectedModelId = $state<string>('');
  let config = $state<InferenceConfig>({ ...DEFAULT_CONFIG });

  let imageQueue = $state<QueuedImage[]>([]);
  let isDragging = $state(false);

  let isRunning = $state(false);
  let isCancelling = $state(false);
  let results = $state<ImageResult[]>([]);
  let progress = $state({ done: 0, total: 0 });
  let elapsedSeconds = $state(0);
  let timerInterval: ReturnType<typeof setInterval> | null = null;
  let timerStart = 0;

  let showAdvanced = $state(false);
  let showCrop = $state(false);

  // ── Derived ──────────────────────────────────────────────────────────────────
  let selectedModel = $derived(availableModels.find(m => m.id === selectedModelId));
  let totalPlaques = $derived(results.reduce((s, r) => s + r.count, 0));
  let errorCount = $derived(results.filter(r => r.error).length);
  let progressPct = $derived(progress.total > 0 ? (progress.done / progress.total) * 100 : 0);
  let canRun = $derived(selectedModel !== undefined && imageQueue.length > 0 && !isRunning);
  let hasResults = $derived(results.length > 0);

  // ── Event listeners ──────────────────────────────────────────────────────────
  let unlisteners: UnlistenFn[] = [];

  onMount(async () => {
    const saved = localStorage.getItem('plaqseg-theme') as 'dark' | 'light' | null;
    if (saved) theme = saved;

    try {
      const { resourceDir } = await import('@tauri-apps/api/path');
      const appDataDir = await resourceDir().catch(() => '');
      if (appDataDir) {
        modelsFolder = appDataDir + '/models';
        await rescanModels();
      }
    } catch { /* no default — user picks */ }

    const unlistenProgress = await listen<ProgressEvent>('inference_progress', (event) => {
      handleProgressEvent(event.payload);
    });
    unlisteners.push(unlistenProgress);

    const unlistenDrop = await listen<{ paths: string[] }>('tauri://drag-drop', (event) => {
      isDragging = false;
      const imgExts = /\.(jpg|jpeg|png|tiff?|bmp)$/i;
      const paths = (event.payload.paths ?? []).filter((p: string) => imgExts.test(p));
      if (paths.length) addPaths(paths);
    });
    unlisteners.push(unlistenDrop);

    const u1 = await listen('tauri://drag-enter', () => { isDragging = true; });
    const u2 = await listen('tauri://drag-over', () => { isDragging = true; });
    const u3 = await listen('tauri://drag-leave', () => { isDragging = false; });
    unlisteners.push(u1, u2, u3);
  });

  onDestroy(() => {
    unlisteners.forEach(fn => fn());
    stopTimer();
  });

  // ── Progress handling ─────────────────────────────────────────────────────────
  function handleProgressEvent(ev: ProgressEvent) {
    if (ev.type === 'image_start') {
      progress.total = ev.total;
      imageQueue = imageQueue.map(q =>
        q.path === ev.image ? { ...q, status: 'running' } : q
      );
    } else if (ev.type === 'image_done') {
      progress.done = ev.index + 1;
      imageQueue = imageQueue.map(q =>
        q.path === ev.image
          ? { ...q, status: 'done', count: ev.count, duration_s: ev.duration_s }
          : q
      );
    } else if (ev.type === 'image_error') {
      progress.done = ev.index + 1;
      imageQueue = imageQueue.map(q =>
        q.path === ev.image
          ? { ...q, status: 'error', error: ev.error }
          : q
      );
    } else if (ev.type === 'done') {
      results = ev.results;
      progress.done = progress.total;
      isRunning = false;
      isCancelling = false;
      stopTimer();
    } else if (ev.type === 'cancelled') {
      isRunning = false;
      isCancelling = false;
      stopTimer();
    }
  }

  // ── Timer ─────────────────────────────────────────────────────────────────────
  function startTimer() {
    timerStart = Date.now();
    elapsedSeconds = 0;
    timerInterval = setInterval(() => {
      elapsedSeconds = (Date.now() - timerStart) / 1000;
    }, 250);
  }
  function stopTimer() {
    if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
  }

  // ── Models ────────────────────────────────────────────────────────────────────
  async function rescanModels() {
    if (!modelsFolder) return;
    try {
      availableModels = await commands.scanModels(modelsFolder);
      if (availableModels.length > 0 && !selectedModelId) {
        selectModel(availableModels[0]);
      }
    } catch {
      availableModels = [];
    }
  }

  async function browseModelsFolder() {
    const folder = await commands.selectModelsFolder();
    if (folder) {
      modelsFolder = folder;
      await rescanModels();
    }
  }

  function selectModel(m: ModelInfo) {
    selectedModelId = m.id;
    config.model_path = m.path;
    config.backend = m.backend;
    if (!m.gpu_capable && config.device !== 'cpu') {
      config.device = 'cpu';
    }
  }

  // ── Images ────────────────────────────────────────────────────────────────────
  function pathToName(p: string): string {
    return p.split(/[/\\]/).pop() ?? p;
  }

  function addPaths(paths: string[]) {
    const existing = new Set(imageQueue.map(q => q.path));
    const newItems: QueuedImage[] = paths
      .filter(p => !existing.has(p))
      .map(p => ({
        path: p,
        name: pathToName(p),
        status: 'queued',
        count: null,
        error: null,
        duration_s: null,
      }));
    imageQueue = [...imageQueue, ...newItems];
  }

  async function addImages() {
    const paths = await commands.selectImages();
    if (paths.length) addPaths(paths);
  }

  async function addFolder() {
    const folder = await commands.selectFolder();
    if (folder) {
      const paths = await commands.getImagesInFolder(folder);
      addPaths(paths);
    }
  }

  function clearQueue() {
    imageQueue = [];
    results = [];
    progress = { done: 0, total: 0 };
    elapsedSeconds = 0;
  }

  function removeImage(path: string) {
    imageQueue = imageQueue.filter(q => q.path !== path);
  }

  // ── Drag & Drop ───────────────────────────────────────────────────────────────
  function onDragOver(e: DragEvent) {
    e.preventDefault();
    isDragging = true;
  }
  function onDragLeave(e: DragEvent) {
    if (!(e.currentTarget as Element).contains(e.relatedTarget as Node)) {
      isDragging = false;
    }
  }
  async function onDrop(e: DragEvent) {
    e.preventDefault();
    isDragging = false;
    // Tauri drag-drop event handles actual fs paths
  }

  // ── Run / Cancel ──────────────────────────────────────────────────────────────
  async function runInference() {
    if (!canRun || !selectedModel) return;
    imageQueue = imageQueue.map(q => ({ ...q, status: 'queued', count: null, error: null }));
    results = [];
    progress = { done: 0, total: imageQueue.length };
    isRunning = true;
    startTimer();
    const inferConfig: InferenceConfig = {
      ...config,
      images: imageQueue.map(q => q.path),
    };
    try {
      await commands.runInference(inferConfig);
    } catch (e) {
      isRunning = false;
      stopTimer();
      console.error('run_inference failed:', e);
    }
  }

  async function cancelInference() {
    isCancelling = true;
    await commands.cancelInference();
  }

  // ── Export ────────────────────────────────────────────────────────────────────
  async function exportJson() {
    await commands.saveResultsJson(JSON.stringify(results, null, 2));
  }

  async function exportCsv() {
    let csv = 'image,plaque_count,x,y,width,height,confidence,area,duration_s,error\n';
    for (const r of results) {
      if (r.detections.length === 0) {
        csv += `${r.image},${r.count},,,,,,,${r.duration_s},${r.error ?? ''}\n`;
      } else {
        for (const d of r.detections) {
          csv += `${r.image},${r.count},${d.x},${d.y},${d.width},${d.height},${d.confidence.toFixed(4)},${d.area},${r.duration_s},${r.error ?? ''}\n`;
        }
      }
    }
    await commands.saveResultsCsv(csv);
  }

  // ── Helpers ───────────────────────────────────────────────────────────────────
  function formatTime(s: number): string {
    if (s < 60) return `${s.toFixed(1)}s`;
    const m = Math.floor(s / 60);
    return `${m}m ${Math.floor(s % 60)}s`;
  }

  function statusIcon(status: string): string {
    if (status === 'running') return 'reload-circle';
    if (status === 'done') return 'checkmark-circle';
    if (status === 'error') return 'close-circle';
    return 'ellipse-outline';
  }

  function statusColor(status: string): string {
    if (status === 'running') return 'var(--accent)';
    if (status === 'done') return 'var(--success)';
    if (status === 'error') return 'var(--error)';
    return 'var(--text-muted)';
  }

  let expandedRows = $state<string[]>([]);
  function toggleExpanded(image: string) {
    if (expandedRows.includes(image)) {
      expandedRows = expandedRows.filter(p => p !== image);
    } else {
      expandedRows = [...expandedRows, image];
    }
  }

  let viewerOpen = $state(false);
  let viewerResult = $state<ImageResult | null>(null);
  let viewerImageSrc = $state('');
  let viewerOriginalWidth = $state(0);
  let viewerOriginalHeight = $state(0);
  let viewerLoading = $state(false);

  async function openViewer(result: ImageResult) {
    viewerResult = result;
    viewerOpen = true;
    viewerLoading = true;
    try {
      const [bytes, w, h] = await commands.readImagePreview(result.image);
      const array = new Uint8Array(bytes);
      const blob = new Blob([array], { type: 'image/jpeg' });
      if (viewerImageSrc) URL.revokeObjectURL(viewerImageSrc);
      viewerImageSrc = URL.createObjectURL(blob);
      viewerOriginalWidth = w;
      viewerOriginalHeight = h;
    } catch(e) {
      console.error(e);
    } finally {
      viewerLoading = false;
    }
  }

  function closeViewer() {
    viewerOpen = false;
    viewerResult = null;
    if (viewerImageSrc) URL.revokeObjectURL(viewerImageSrc);
    viewerImageSrc = '';
  }
</script>

<svelte:head>
  <title>PlaqSeg Desktop</title>
</svelte:head>

<div class="app-shell">

  <!-- ── Navbar ──────────────────────────────────────────────────────────────── -->
  <nav class="navbar">
    <div class="navbar-logo">
      <div class="logo-icon">
        <ion-icon name="flask" style="font-size:14px;color:#fff;"></ion-icon>
      </div>
      PlaqSeg Desktop
    </div>
    <div class="navbar-spacer"></div>
    <div class="navbar-meta">
      {#if availableModels.length > 0}
        <span>{availableModels.length} model{availableModels.length !== 1 ? 's' : ''} loaded</span>
      {/if}
      {#if imageQueue.length > 0}
        <span>{imageQueue.length} image{imageQueue.length !== 1 ? 's' : ''} queued</span>
      {/if}
    </div>
    <button class="theme-toggle" onclick={() => theme = theme === 'dark' ? 'light' : 'dark'}
      title="Toggle theme">
      <ion-icon name={theme === 'dark' ? 'sunny' : 'moon'}></ion-icon>
    </button>
  </nav>

  <!-- ══════════════ LEFT PANEL — Configuration ════════════════════════════════ -->
  <aside class="panel panel-left">

    <!-- Models folder -->
    <div class="section-header">
      <span class="section-title">Models</span>
      <button class="btn btn-ghost btn-sm" onclick={rescanModels} title="Rescan models folder">
        <ion-icon name="refresh"></ion-icon> Scan
      </button>
    </div>
    <div class="path-row">
      <div class="path-display" title={modelsFolder || 'No folder selected'}>
        {modelsFolder || 'No models folder selected'}
      </div>
      <button class="btn btn-outline btn-sm" onclick={browseModelsFolder}>
        <ion-icon name="folder-open-outline"></ion-icon>
      </button>
    </div>

    <!-- Model cards -->
    {#if availableModels.length === 0}
      <div class="no-models-warning">
        <ion-icon name="warning-outline"></ion-icon>
        No models found. Select a folder containing <code>.onnx</code> files.
      </div>
    {:else}
      {#each availableModels as model (model.id)}
        <div
          class="model-card {selectedModelId === model.id ? 'selected' : ''}"
          onclick={() => selectModel(model)}
          role="button"
          tabindex="0"
          onkeydown={(e) => e.key === 'Enter' && selectModel(model)}
        >
          <div class="model-card-header">
            <span class="model-card-name">{model.label}</span>
            <div class="model-card-badges">
              <span class="badge badge-size">{model.size_mb.toFixed(1)} MB</span>
              <span class="badge badge-onnx">
                {model.backend.toUpperCase()}
              </span>
              {#if model.gpu_capable}
                <span class="badge badge-gpu">
                  <ion-icon name="flash"></ion-icon> GPU
                </span>
              {:else}
                <span class="badge badge-cpu">CPU</span>
              {/if}
            </div>
          </div>
          <div class="model-card-desc">{model.description}</div>
        </div>
      {/each}
    {/if}

    <!-- Device selector -->
    <div class="section-header" style="padding-top: 12px;">
      <span class="section-title">Device</span>
    </div>
    <div class="device-radios">
      <input type="radio" id="dev-auto" class="device-radio" name="device" value="auto"
        bind:group={config.device} />
      <label for="dev-auto" class="device-radio-label">
        <ion-icon name="hardware-chip-outline" class="device-icon"></ion-icon>Auto
      </label>

      <input type="radio" id="dev-cpu" class="device-radio" name="device" value="cpu"
        bind:group={config.device} />
      <label for="dev-cpu" class="device-radio-label">
        <ion-icon name="desktop-outline" class="device-icon"></ion-icon>CPU
      </label>

      <input type="radio" id="dev-cuda" class="device-radio" name="device" value="cuda"
        bind:group={config.device} />
      <label for="dev-cuda" class="device-radio-label" title="NVIDIA CUDA GPU">
        <ion-icon name="flash-outline" class="device-icon"></ion-icon>CUDA
      </label>

      <input type="radio" id="dev-dml" class="device-radio" name="device" value="directml"
        bind:group={config.device} />
      <label for="dev-dml" class="device-radio-label" title="DirectML — AMD/Intel GPU">
        <ion-icon name="hardware-chip-outline" class="device-icon"></ion-icon>DirectML
      </label>
    </div>

    <!-- Workers -->
    <div class="section-header">
      <span class="section-title">Workers</span>
    </div>
    <div class="workers-row">
      <input type="range" min="1" max="32" step="1" bind:value={config.workers} />
      <div class="workers-badge">{config.workers}</div>
    </div>
    <div class="form-group" style="padding-top:0;margin-top:-6px;">
      <span style="font-size:10px;color:var(--text-muted);">
        {config.device === 'cpu' || !selectedModel?.gpu_capable
          ? 'Parallel CPU workers — set to your core count'
          : 'GPU mode: workers ignored (GPU handles parallelism)'}
      </span>
    </div>

    <!-- Advanced collapsible -->
    <div class="collapsible-header" onclick={() => showAdvanced = !showAdvanced}
      role="button" tabindex="0" onkeydown={(e) => e.key === 'Enter' && (showAdvanced = !showAdvanced)}>
      <span class="collapsible-title">Advanced — Tiling &amp; Thresholds</span>
      <ion-icon name="chevron-down" class="collapsible-arrow {showAdvanced ? 'open' : ''}"></ion-icon>
    </div>
    {#if showAdvanced}
      <div class="collapsible-content" style="max-height:500px;">
        <div class="form-group" style="padding-top:10px;">
          <div class="form-label">Tile size (px) <span class="form-label-value">{config.tile_size}</span></div>
          <input type="range" min="320" max="2560" step="64" bind:value={config.tile_size} />
        </div>
        <div class="form-group">
          <div class="form-label">Stride (px) <span class="form-label-value">{config.stride}</span></div>
          <input type="range" min="160" max="2048" step="64" bind:value={config.stride} />
          <div style="font-size:10px;color:var(--text-muted);margin-top:3px;">Overlap = {config.tile_size - config.stride}px</div>
        </div>
        <div class="form-group">
          <div class="form-label">Confidence <span class="form-label-value">{config.conf_threshold.toFixed(2)}</span></div>
          <input type="range" min="0.01" max="0.99" step="0.01" bind:value={config.conf_threshold} />
        </div>
        <div class="form-group">
          <div class="form-label">NMS IoU <span class="form-label-value">{config.nms_threshold.toFixed(2)}</span></div>
          <input type="range" min="0.1" max="0.9" step="0.05" bind:value={config.nms_threshold} />
        </div>
      </div>
    {/if}

    <!-- Crop collapsible -->
    <div class="collapsible-header" onclick={() => showCrop = !showCrop}
      role="button" tabindex="0" onkeydown={(e) => e.key === 'Enter' && (showCrop = !showCrop)}>
      <span class="collapsible-title">Petri Dish Crop</span>
      <ion-icon name="chevron-down" class="collapsible-arrow {showCrop ? 'open' : ''}"></ion-icon>
    </div>
    {#if showCrop}
      <div class="collapsible-content" style="max-height:500px;">
        <div class="form-group" style="padding-top:10px;">
          <label class="toggle-wrap">
            <div class="toggle">
              <input type="checkbox" bind:checked={config.crop_enabled} />
              <div class="toggle-slider"></div>
            </div>
            <span style="font-size:12px;color:var(--text-secondary);">Auto-crop to Petri dish</span>
          </label>
        </div>
        {#if config.crop_enabled}
          <div class="form-group">
            <div class="form-label">Scale <span class="form-label-value">{config.crop_scale.toFixed(2)}</span></div>
            <input type="range" min="0.05" max="1.0" step="0.05" bind:value={config.crop_scale} />
          </div>
          <div class="form-group">
            <div class="form-label">Hough param1 <span class="form-label-value">{config.crop_param1}</span></div>
            <input type="range" min="10" max="300" step="5" bind:value={config.crop_param1} />
          </div>
          <div class="form-group">
            <div class="form-label">Hough param2 <span class="form-label-value">{config.crop_param2}</span></div>
            <input type="range" min="5" max="100" step="5" bind:value={config.crop_param2} />
          </div>
          <div class="form-group">
            <div class="form-label">Padding (px) <span class="form-label-value">{config.crop_padding}</span></div>
            <input type="range" min="0" max="100" step="5" bind:value={config.crop_padding} />
          </div>
        {/if}
      </div>
    {/if}

  </aside>

  <!-- ══════════════ CENTER — Image Queue ══════════════════════════════════════ -->
  <main class="center-panel">

    <!-- Drop zone -->
    <div
      class="drop-zone {isDragging ? 'dragging' : ''}"
      ondragover={onDragOver}
      ondragleave={onDragLeave}
      ondrop={onDrop}
      role="region"
      aria-label="Image drop zone"
    >
      <ion-icon name="images-outline" class="drop-icon"></ion-icon>
      <div class="drop-title">Drop images here</div>
      <div class="drop-hint">JPEG · PNG · TIFF · BMP — files or an entire folder</div>
      <div class="drop-actions">
        <button class="btn btn-outline" onclick={addImages} disabled={isRunning}>
          <ion-icon name="document-outline"></ion-icon> Add Images
        </button>
        <button class="btn btn-outline" onclick={addFolder} disabled={isRunning}>
          <ion-icon name="folder-outline"></ion-icon> Add Folder
        </button>
      </div>
    </div>

    <!-- Queue header -->
    {#if imageQueue.length > 0}
      <div class="queue-header">
        <span class="queue-count">
          {imageQueue.length} image{imageQueue.length !== 1 ? 's' : ''}
          {#if isRunning}&nbsp;· {progress.done}/{progress.total} done{/if}
        </span>
        <button class="btn btn-ghost btn-sm" onclick={clearQueue} disabled={isRunning}>
          <ion-icon name="trash-outline"></ion-icon> Clear all
        </button>
      </div>

      <div class="queue-list">
        {#each imageQueue as item (item.path)}
          <div class="queue-item status-{item.status} fade-in">
            <ion-icon
              name={statusIcon(item.status)}
              class={item.status === 'running' ? 'spin' : ''}
              style="font-size:16px;flex-shrink:0;color:{statusColor(item.status)};"
            ></ion-icon>
            <div class="queue-item-info">
              <div class="queue-item-name" title={item.path}>{item.name}</div>
              <div class="queue-item-meta">
                {#if item.status === 'running'}
                  Processing…
                {:else if item.status === 'done' && item.duration_s !== null}
                  {item.duration_s.toFixed(2)}s
                {:else if item.status === 'error'}
                  {item.error ?? 'Error'}
                {:else}
                  {item.path.split(/[/\\]/).slice(-2, -1)[0] ?? ''}
                {/if}
              </div>
            </div>
            {#if item.status === 'done' && item.count !== null}
              <span class="queue-item-count">{item.count}</span>
            {:else if item.status === 'error'}
              <span class="badge badge-error">Error</span>
            {:else if item.status === 'running'}
              <span class="badge badge-running">Running</span>
            {:else}
              <span class="badge badge-queued">Queued</span>
            {/if}
            {#if !isRunning}
              <button class="btn btn-ghost btn-sm"
                onclick={() => removeImage(item.path)}
                style="padding:2px;color:var(--text-muted);">
                <ion-icon name="close-outline"></ion-icon>
              </button>
            {/if}
          </div>
        {/each}
      </div>

    {:else}
      <div class="empty-state">
        <ion-icon name="images-outline" class="empty-state-icon"></ion-icon>
        <div class="empty-state-text">No images queued — drop some above or use the buttons</div>
      </div>
    {/if}
  </main>

  <!-- ══════════════ RIGHT PANEL — Run &amp; Results ═══════════════════════════ -->
  <aside class="panel panel-right" style="display:flex;flex-direction:column;">

    <!-- Run / Cancel -->
    <div style="padding:14px;">
      {#if !isRunning}
        <button class="btn-run" onclick={runInference} disabled={!canRun}>
          {#if !selectedModel}
            <ion-icon name="warning-outline"></ion-icon> Select a model
          {:else if imageQueue.length === 0}
            <ion-icon name="warning-outline"></ion-icon> Add images first
          {:else}
            <ion-icon name="play"></ion-icon> Run Inference
          {/if}
        </button>
      {:else}
        <button class="btn-run running" onclick={cancelInference} disabled={isCancelling}>
          {#if isCancelling}
            <ion-icon name="reload" class="spin"></ion-icon> Cancelling…
          {:else}
            <ion-icon name="stop"></ion-icon> Cancel
          {/if}
        </button>
      {/if}
    </div>

    <!-- Progress -->
    {#if isRunning || (hasResults && progress.total > 0)}
      <div style="padding:0 14px 12px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
          <span style="font-size:11px;font-weight:600;color:var(--text-secondary);">
            {isRunning ? 'Processing…' : 'Complete'}
          </span>
          <span style="font-size:11px;color:var(--text-muted);font-variant-numeric:tabular-nums;">
            {Math.round(progressPct)}%
            {#if isRunning}&nbsp;· {formatTime(elapsedSeconds)}{/if}
          </span>
        </div>
        <div class="progress-wrap">
          <div class="progress-fill" style="width:{progressPct}%"></div>
        </div>
        <div style="margin-top:6px;font-size:10px;color:var(--text-muted);">
          {progress.done} / {progress.total} images
        </div>
      </div>
    {/if}

    <!-- Stats -->
    {#if hasResults}
      <div class="stats-row">
        <div class="stat-card">
          <div class="stat-value">{results.length}</div>
          <div class="stat-label">Images</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" style="color:var(--success)">{totalPlaques}</div>
          <div class="stat-label">Plaques</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" style="color:{errorCount > 0 ? 'var(--error)' : 'var(--text-muted)'}">
            {errorCount}
          </div>
          <div class="stat-label">Errors</div>
        </div>
      </div>
      <div style="padding:0 14px 8px;">
        <div style="font-size:11px;color:var(--text-muted);">
          Total: {formatTime(elapsedSeconds)}
          · Avg: {results.length > 0 ? (elapsedSeconds / results.length).toFixed(2) : '—'}s/img
        </div>
      </div>
    {:else if !isRunning}
      <div class="empty-state" style="padding:24px 20px;">
        <ion-icon name="bar-chart-outline" class="empty-state-icon"></ion-icon>
        <div class="empty-state-text">Results appear here after inference</div>
      </div>
    {/if}

    <!-- Results table -->
    {#if hasResults}
      <div class="section-header" style="padding-bottom:6px;">
        <span class="section-title">Per-Image Results</span>
      </div>
      <div class="results-table-wrap">
        <table class="results-table">
          <thead>
            <tr>
              <th>Image</th>
              <th>Plaques</th>
              <th>Time</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {#each results as r (r.image)}
              <tr onclick={() => toggleExpanded(r.image)} style="cursor:pointer;" class="results-row-clickable">
                <td title={r.image}>
                  <ion-icon name={expandedRows.includes(r.image) ? "chevron-down" : "chevron-forward"} style="vertical-align:middle;margin-right:2px;font-size:10px;"></ion-icon>
                  {r.image.split(/[/\\]/).pop()}
                </td>
                <td style="font-weight:700;color:var(--success);font-variant-numeric:tabular-nums;">{r.count}</td>
                <td style="font-variant-numeric:tabular-nums;">{r.duration_s.toFixed(2)}s</td>
                <td>
                  <button class="btn btn-ghost btn-sm" onclick={(e) => { e.stopPropagation(); openViewer(r); }} style="padding: 2px 6px; margin-right: 4px;" title="View Image Overlay">
                    <ion-icon name="image-outline"></ion-icon>
                  </button>
                  {#if r.error}
                    <ion-icon name="close-circle" style="color:var(--error)" title={r.error}></ion-icon>
                  {:else}
                    <ion-icon name="checkmark-circle" style="color:var(--success)"></ion-icon>
                  {/if}
                </td>
              </tr>
              {#if expandedRows.includes(r.image) && r.count > 0}
                <tr>
                  <td colspan="4" style="padding: 0; background-color: var(--surface-2); border-bottom: 1px solid var(--border-subtle);">
                    <div style="padding: 8px 12px; font-size: 11px;">
                      <div style="font-weight: 600; color: var(--text-secondary); margin-bottom: 4px;">Detections ({r.count})</div>
                      <div style="max-height: 150px; overflow-y: auto;">
                        <table style="width: 100%; text-align: left; border-collapse: collapse;">
                          <thead style="position: sticky; top: 0; background: var(--surface-2);">
                            <tr>
                              <th style="padding-bottom: 4px; border-bottom: 1px solid var(--border-subtle); color: var(--text-muted); font-weight: normal;">Position (x,y)</th>
                              <th style="padding-bottom: 4px; border-bottom: 1px solid var(--border-subtle); color: var(--text-muted); font-weight: normal;">Size (w×h)</th>
                              <th style="padding-bottom: 4px; border-bottom: 1px solid var(--border-subtle); color: var(--text-muted); font-weight: normal;">Confidence</th>
                            </tr>
                          </thead>
                          <tbody>
                            {#each r.detections as d}
                              <tr>
                                <td style="padding: 4px 0; border-bottom: 1px dashed var(--border-subtle); font-variant-numeric: tabular-nums;">{d.x}, {d.y}</td>
                                <td style="padding: 4px 0; border-bottom: 1px dashed var(--border-subtle); font-variant-numeric: tabular-nums;">{d.width}×{d.height}</td>
                                <td style="padding: 4px 0; border-bottom: 1px dashed var(--border-subtle); font-variant-numeric: tabular-nums;">{(d.confidence * 100).toFixed(1)}%</td>
                              </tr>
                            {/each}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </td>
                </tr>
              {/if}
            {/each}
          </tbody>
        </table>
      </div>
    {/if}

    <div style="flex:1;"></div>

    <!-- Export -->
    {#if hasResults}
      <div class="export-buttons">
        <div class="export-row">
          <button class="btn btn-outline" onclick={exportJson}>
            <ion-icon name="code-download-outline"></ion-icon> Export JSON
          </button>
          <button class="btn btn-outline" onclick={exportCsv}>
            <ion-icon name="document-text-outline"></ion-icon> Export CSV
          </button>
        </div>
      </div>
    {/if}

    <!-- Model summary footer -->
    {#if selectedModel}
      <div style="padding:10px 14px;border-top:1px solid var(--border-subtle);font-size:10px;color:var(--text-muted);">
        <strong style="color:var(--text-secondary)">{selectedModel.label}</strong>
        · {selectedModel.backend.toUpperCase()}
        · {config.device.toUpperCase()}
        · {config.workers}W
        · Tile {config.tile_size}px / Stride {config.stride}px
        · Conf {config.conf_threshold.toFixed(2)}
        · NMS {config.nms_threshold.toFixed(2)}
      </div>
    {/if}

  </aside>
</div>

<!-- Viewer Modal -->
{#if viewerOpen && viewerResult}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="modal-backdrop" onclick={closeViewer}>
    <div class="modal-content" onclick={(e) => e.stopPropagation()}>
      <div class="modal-header">
        <div style="font-weight: 600; font-size: 14px;">{viewerResult.image.split(/[/\\]/).pop()}</div>
        <button class="btn btn-ghost" onclick={closeViewer} style="padding: 4px;">
          <ion-icon name="close-outline" style="font-size: 20px;"></ion-icon>
        </button>
      </div>
      <div class="modal-body">
        {#if viewerLoading}
          <div style="display: flex; justify-content: center; align-items: center; height: 300px;">
            <ion-icon name="reload" class="spin" style="font-size: 32px; color: var(--accent);"></ion-icon>
          </div>
        {:else if viewerImageSrc}
          <div class="image-overlay-container">
            <img src={viewerImageSrc} alt="Preview" class="preview-img" />
            {#each viewerResult.detections as d}
              <div class="bbox" 
                   style="left: {(d.x / viewerOriginalWidth) * 100}%; 
                          top: {(d.y / viewerOriginalHeight) * 100}%; 
                          width: {(d.width / viewerOriginalWidth) * 100}%; 
                          height: {(d.height / viewerOriginalHeight) * 100}%"
                   title="Conf: {(d.confidence*100).toFixed(1)}%">
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </div>
  </div>
{/if}
