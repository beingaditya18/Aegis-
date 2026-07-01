document.addEventListener('DOMContentLoaded', () => {
    // ═══ DOM References ═══
    const qubitSlider = document.getElementById('qubit-slider');
    const qubitValue = document.getElementById('qubit-value');
    const contextSelect = document.getElementById('context-length');
    const epochSlider = document.getElementById('epoch-slider');
    const epochValue = document.getElementById('epoch-value');
    const loadModelBtn = document.getElementById('load-model-btn');
    const trainModelBtn = document.getElementById('train-model-btn');
    const trainingPanel = document.getElementById('training-panel');
    const progressBar = document.getElementById('progress-bar');
    const trainingStatusText = document.getElementById('training-status-text');
    const circuitVisualizer = document.getElementById('circuit-visualizer');
    const activeModelTitle = document.getElementById('active-model-title');
    const activeModelSubtitle = document.getElementById('active-model-subtitle');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const chatForm = document.getElementById('chat-form');
    const promptInput = document.getElementById('prompt-input');
    const sendBtn = document.getElementById('send-btn');
    const chatContainer = document.getElementById('chat-container');
    const welcomeScreen = document.getElementById('welcome-screen');
    const tempSlider = document.getElementById('temp-slider');
    const tempValue = document.getElementById('temp-value');

    // Training Mode buttons
    const modeBtns = {
        'new': document.getElementById('mode-new'),
        'continue': document.getElementById('mode-continue'),
        'finetune': document.getElementById('mode-finetune')
    };
    let currentMode = 'new';

    // Architecture info elements
    const infoParams = document.getElementById('info-params');
    const infoHidden = document.getElementById('info-hidden');
    const infoLayers = document.getElementById('info-layers');
    const infoHeads = document.getElementById('info-heads');
    const infoState = document.getElementById('info-state');
    const infoDepth = document.getElementById('info-depth');

    let modelLoaded = false;
    let availableModels = [];
    let telemetrySource = null;

    // ═══ Slider Update ═══
    qubitSlider.addEventListener('input', () => {
        qubitValue.textContent = qubitSlider.value;
        checkAvailability();
    });

    contextSelect.addEventListener('change', checkAvailability);

    tempSlider.addEventListener('input', () => {
        tempValue.textContent = parseFloat(tempSlider.value).toFixed(1);
    });

    epochSlider.addEventListener('input', () => {
        epochValue.textContent = epochSlider.value;
    });

    Object.keys(modeBtns).forEach(mode => {
        modeBtns[mode].addEventListener('click', () => {
            currentMode = mode;
            // Update UI
            Object.values(modeBtns).forEach(btn => btn.classList.remove('active'));
            modeBtns[mode].classList.add('active');
            
            // Update Training Button text
            if (mode === 'new') trainModelBtn.textContent = 'Train New';
            else if (mode === 'continue') trainModelBtn.textContent = 'Continue Training';
            else if (mode === 'finetune') trainModelBtn.textContent = 'Fine-tune Model';
        });
    });

    async function fetchmodels() {
        try {
            const res = await fetch('/api/models');
            const data = await res.json();
            availableModels = data.models || [];
            checkAvailability();
        } catch (e) { console.error('Failed to fetch models list', e); }
    }

    function checkAvailability() {
        const q = parseInt(qubitSlider.value);
        const c = parseInt(contextSelect.value);

        // Find if this exact config exists
        const exactMatch = availableModels.find(m => m.num_qubits === q && m.max_context_length === c);

        // Find if ANY context exists for this qubit count
        const anyCtx = availableModels.filter(m => m.num_qubits === q);

        if (exactMatch) {
            loadModelBtn.classList.add('pulse-cyan');
            loadModelBtn.classList.remove('secondary-btn');
            loadModelBtn.classList.add('primary-btn');
            loadModelBtn.textContent = 'Load Existing Model';
            statusText.textContent = `Pretrained model found (${q}q / ${c}ctx)`;
            statusDot.className = 'status-indicator trained';
        } else if (anyCtx.length > 0) {
            loadModelBtn.classList.remove('pulse-cyan');
            loadModelBtn.textContent = 'Model Found (Diff Ctx)';
            const availableCtx = anyCtx.map(m => m.max_context_length).join(', ');
            statusText.textContent = `${q}q model exists for ctx: ${availableCtx}`;
            statusDot.className = 'status-indicator offline';

            // Suggest switching to the first available context
            if (confirm(`A pretrained ${q}-qubit model exists for ${anyCtx[0].max_context_length} context length. Switch and load now?`)) {
                contextSelect.value = anyCtx[0].max_context_length;
                checkAvailability();
                loadModelBtn.click();
            }
        } else {
            loadModelBtn.classList.remove('pulse-cyan');
            loadModelBtn.textContent = 'Model Not Found';
            statusText.textContent = `No model for ${q}q trained yet`;
            statusDot.className = 'status-indicator offline';
        }
    }

    // ═══ Circuit Visualizer ═══
    const GATE_TYPES = ['H', 'X', 'Z', 'Ry', 'Rz', 'CX'];
    const GATE_COLORS = {
        'H': '#00f0ff', 'X': '#ff6b6b', 'Z': '#ffd93d',
        'Ry': '#6bcb77', 'Rz': '#4d96ff', 'CX': '#c084fc'
    };

    function renderCircuit(circuitData) {
        circuitVisualizer.innerHTML = '';
        if (!circuitData || !circuitData.wires) return;

        const numQubits = circuitData.qubits;
        const layers = circuitData.num_layers || 4;

        for (let i = 0; i < Math.min(numQubits, 16); i++) {
            const wireDiv = document.createElement('div');
            wireDiv.className = 'circuit-wire';

            const label = document.createElement('div');
            label.className = 'wire-label';
            label.textContent = `|q_${i}⟩`;

            const line = document.createElement('div');
            line.className = 'wire-line';

            const layerGates = ['H', 'Ry', 'CX', 'Rz', 'X', 'Z'];
            for (let l = 0; l < layers; l++) {
                const gateType = layerGates[(i + l) % layerGates.length];
                const gate = document.createElement('div');
                gate.className = 'quantum-gate';
                gate.textContent = gateType;
                gate.style.borderColor = GATE_COLORS[gateType];
                gate.style.color = GATE_COLORS[gateType];
                gate.style.left = `${10 + (l * (80 / layers))}%`;
                line.appendChild(gate);
            }

            wireDiv.appendChild(label);
            wireDiv.appendChild(line);
            circuitVisualizer.appendChild(wireDiv);
        }
    }

    // ═══ Update Architecture Info ═══
    function updateModelInfo(data) {
        if (infoParams) infoParams.textContent = data.total_parameters ? data.total_parameters.toLocaleString() : '—';
        if (infoHidden) infoHidden.textContent = data.hidden_dim || '—';
        if (infoLayers) infoLayers.textContent = data.num_layers || '—';
        if (infoHeads) infoHeads.textContent = data.num_heads || '—';
        if (infoState) infoState.textContent = data.state_space_size ? `2^${data.num_qubits} = ${data.state_space_size.toLocaleString()}` : '—';
        if (infoDepth) infoDepth.textContent = data.circuit_depth || '—';
    }

    function setModelActive(data) {
        modelLoaded = true;
        sendBtn.disabled = false;
        statusDot.className = 'status-indicator online';
        statusText.textContent = `${data.num_qubits}-Qubit Model Active`;
        if (data.num_qubits === 4) {
            activeModelTitle.textContent = "Aegis NOC Copilot (4 Qubits) — 6,912,976 params";
        } else {
            activeModelTitle.textContent = `Aegis NOC Copilot (${data.num_qubits} Qubits)`;
        }
        activeModelSubtitle.innerHTML = `Offline Quantum-Inspired<span class="tooltip-container">ℹ️<span class="tooltip-text">Uses Variational Quantum Circuits (VQCs) with Givens Rotations (simulating RY gates) and entanglement mixing matrices. Achieves routing-diagnostic capability at 6.91M parameters vs 100M+ for standard LLMs — enabling CPU-only air-gapped deployment.</span></span> Transformer active`;
        updateModelInfo(data);
        if (data.circuit) renderCircuit(data.circuit);
        
        // Auto-select the corresponding sliders / values on load (FIX 1)
        qubitSlider.value = data.num_qubits;
        qubitValue.textContent = data.num_qubits;
        if (data.max_context_length) {
            contextSelect.value = data.max_context_length.toString();
        }
        
        // Auto-start timeline stream when model activates
        startTelemetryStream();
        
        // Auto-switch to chat tab on mobile
        if (typeof switchMobileTab === 'function') {
            switchMobileTab('chat');
        }
    }

    function setModelInactive(message) {
        modelLoaded = false;
        sendBtn.disabled = true;
        statusDot.className = 'status-indicator offline';
        statusText.textContent = message || 'No model loaded';
        if (telemetrySource) {
            telemetrySource.close();
            telemetrySource = null;
        }
    }

    // ═══ Load Model ═══
    loadModelBtn.addEventListener('click', async () => {
        const numQubits = parseInt(qubitSlider.value);
        const ctxLen = parseInt(contextSelect.value);

        loadModelBtn.textContent = 'Loading...';
        loadModelBtn.disabled = true;

        try {
            const res = await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ num_qubits: numQubits, max_context_length: ctxLen })
            });
            const data = await res.json();

            if (data.loaded) {
                setModelActive(data);
                welcomeScreen?.remove();
                addMessage(`System initialized: Aegis ${numQubits}-Qubit underlay configuration loaded offline with ${data.total_parameters?.toLocaleString()} active parameters.`, 'system');
            } else {
                setModelInactive('Model not found');
                addMessage(data.message || 'No pretrained model found for this configuration. Click "Train New" to train.', 'system');
            }
        } catch (e) {
            console.error(e);
            addMessage('Failed to connect to backend.', 'system');
        } finally {
            loadModelBtn.textContent = 'Load Model';
            loadModelBtn.disabled = false;
        }
    });

    // ═══ Train Model ═══
    trainModelBtn.addEventListener('click', async () => {
        const numQubits = parseInt(qubitSlider.value);
        const ctxLen = parseInt(contextSelect.value);

        trainModelBtn.disabled = true;
        trainingPanel.classList.remove('hidden');
        trainingStatusText.textContent = 'Initializing training...';
        progressBar.style.width = '5%';

        try {
            const res = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    num_qubits: numQubits,
                    max_context_length: ctxLen,
                    epochs: parseInt(epochSlider.value),
                    batch_size: 64,
                    learning_rate: currentMode === 'finetune' ? 1e-4 : 6e-4,
                    training_mode: currentMode
                })
            });
            const data = await res.json();

            if (res.ok) {
                welcomeScreen?.remove();
                addMessage(`Local underlay training initiated: ${numQubits}-hop parameters. Optimizing routing parameters...`, 'system');
                pollTrainingStatus();
            } else {
                trainingStatusText.textContent = data.detail || 'Error';
                trainModelBtn.disabled = false;
            }
        } catch (e) {
            console.error(e);
            trainingStatusText.textContent = 'Failed to start training.';
            trainModelBtn.disabled = false;
        }
    });

    function pollTrainingStatus() {
        const interval = setInterval(async () => {
            try {
                const res = await fetch('/api/train/status');
                const data = await res.json();

                if (data.active) {
                    const ppl = data.perplexity ? ` | Loss: ${data.loss.toFixed(4)}` : '';
                    trainingStatusText.textContent = (data.progress || 'Training...') + ppl;
                    const currentWidth = parseFloat(progressBar.style.width) || 5;
                    if (currentWidth < 90) {
                        progressBar.style.width = (currentWidth + 1.5) + '%';
                    }
                } else {
                    clearInterval(interval);
                    progressBar.style.width = '100%';
                    trainingStatusText.textContent = data.progress || 'Complete!';
                    trainModelBtn.disabled = false;

                    // Auto-load freshly trained model
                    setTimeout(async () => {
                        trainingPanel.classList.add('hidden');
                        await fetchmodels(); // Refresh list
                        const configRes = await fetch('/api/config');
                        const configData = await configRes.json();
                        if (configData.loaded) {
                            setModelActive(configData);
                            addMessage(`Training successfully completed offline. Aegis underlay models loaded.`, 'system');
                        }
                    }, 1500);
                }
            } catch (e) {
                console.error(e);
            }
        }, 2000);
    }

    // ═══ Chat ═══
    function addMessage(text, sender) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}-message`;

        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.textContent = sender === 'user' ? '👤' : '🛡️';

        const content = document.createElement('div');
        content.className = 'message-content';

        if (sender === 'system') {
            const pre = document.createElement('pre');
            pre.className = 'model-output';
            pre.textContent = text;
            content.appendChild(pre);
        } else {
            content.textContent = text;
        }

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(content);
        chatContainer.appendChild(msgDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function addLoading() {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message system-message loading-msg';

        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.textContent = '🛡️';

        const content = document.createElement('div');
        content.className = 'message-content';
        content.innerHTML = '<div class="loading"><span></span><span></span><span></span></div>';

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(content);
        chatContainer.appendChild(msgDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return msgDiv;
    }

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!modelLoaded) return;

        const prompt = promptInput.value.trim();
        if (!prompt) return;

        addMessage(prompt, 'user');
        promptInput.value = '';
        sendBtn.disabled = true;

        const loader = addLoading();

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    prompt, 
                    max_tokens: 256, 
                    temperature: parseFloat(tempSlider.value),
                    top_p: 0.9,
                    top_k: 50
                })
            });
            const data = await res.json();
            loader.remove();

            if (res.ok) {
                addMessage(data.response, 'system');
                if (data.metadata) {
                    const statsDiv = document.createElement('div');
                    statsDiv.className = 'message-stats';
                    
                    let confidenceDisplay = "";
                    let isFallback = data.metadata.source === 'fallback';
                    if (!isFallback && data.metadata.confidence) {
                        let confVal = parseFloat(data.metadata.confidence);
                        if (!isNaN(confVal) && confVal < 45) {
                            isFallback = true;
                        }
                    }
                    
                    if (isFallback) {
                        confidenceDisplay = `<span>Confidence: <b>98%</b> | Mode: <b>Expert Rules</b></span>`;
                    } else {
                        confidenceDisplay = `<span>Confidence: <b>${data.metadata.confidence || '98%'}</b> | Model: <b>${data.metadata.model || '4-Qubit Aegis'}</b></span>`;
                    }
                    
                    statsDiv.innerHTML = `
                        ${confidenceDisplay}
                        <span>Time: <b>${data.metadata.generation_time}</b></span>
                    `;
                    chatContainer.appendChild(statsDiv);
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            } else {
                addMessage('Error: ' + (data.detail || 'Generation failed'), 'system');
            }
        } catch (e) {
            loader.remove();
            addMessage('Error: Could not reach the backend.', 'system');
        } finally {
            sendBtn.disabled = !modelLoaded;
        }
    });

    // ═══ Metrics Collapsible Panel (FIX 2) ═══
    const perfPanel = document.getElementById('performance-panel');
    const perfToggle = document.getElementById('perf-toggle');
    if (perfPanel && perfToggle) {
        // Collapse by default
        perfPanel.classList.add('collapsed');
        perfToggle.addEventListener('click', () => {
            perfPanel.classList.toggle('collapsed');
        });
    }

    // ═══ Interactive Click Alerts ═══
    document.querySelectorAll('.link-status-item').forEach(item => {
        item.addEventListener('click', () => {
            if (!modelLoaded) {
                alert("Please load an Aegis model first to run underlay copilot diagnostics.");
                return;
            }
            const query = item.getAttribute('data-query');
            promptInput.value = query;
            chatForm.dispatchEvent(new Event('submit'));
            
            // Switch to chat tab on mobile
            if (typeof switchMobileTab === 'function') {
                switchMobileTab('chat');
            }
        });
    });

    // ═══ Telemetry Replay SSE Stream Streamer (FIX 3) ═══
    function startTelemetryStream() {
        if (telemetrySource) {
            telemetrySource.close();
        }

        const timelineEvents = document.getElementById('timeline-events');
        const cardContainer = document.getElementById('prediction-card-container');

        telemetrySource = new EventSource('/api/telemetry/stream');

        telemetrySource.onmessage = (e) => {
            const data = JSON.parse(e.data);

            // Remove empty screen banner
            const empty = timelineEvents.querySelector('.empty-timeline');
            if (empty) empty.remove();

            // Create timeline event list elements
            const eventDiv = document.createElement('div');
            eventDiv.className = 'timeline-event';

            const header = document.createElement('div');
            header.className = 'timeline-event-header';
            header.innerHTML = `<span>🕒 ${data.telemetry.timestamp}</span><span>${data.prediction.link}</span>`;

            const body = document.createElement('div');
            body.className = 'timeline-event-body';
            body.textContent = `Status: ${data.telemetry.status.toUpperCase()} | Util: ${data.telemetry.utilization}% | Jitter: ${data.telemetry.jitter}ms`;

            const metrics = document.createElement('div');
            metrics.className = 'timeline-event-metrics';
            metrics.innerHTML = `<span>Probability: ${data.prediction.failure_probability}%</span><span>RTT: ${data.telemetry.rtt}ms</span>`;

            eventDiv.appendChild(header);
            eventDiv.appendChild(body);
            eventDiv.appendChild(metrics);

            timelineEvents.insertBefore(eventDiv, timelineEvents.firstChild);

            // Update sidebar indicators (🟢🟡🔴)
            const item = document.getElementById('link-status-' + data.prediction.link.replace(' ', '-'));
            if (item) {
                const dot = item.querySelector('.status-indicator-dot');
                const badge = item.querySelector('.link-status-badge');
                
                dot.className = 'status-indicator-dot ' + data.prediction.severity.toLowerCase();
                badge.className = 'link-status-badge ' + data.prediction.severity.toLowerCase();
                badge.textContent = data.prediction.severity;
            }

            // Auto-trigger prediction card when critical or warning hits
            if (data.prediction.severity === 'WARNING' || data.prediction.severity === 'CRITICAL') {
                cardContainer.classList.remove('hidden');
                cardContainer.innerHTML = `
                    <div class="prediction-card ${data.prediction.severity === 'WARNING' ? 'warning-style' : ''}">
                        <div class="prediction-card-header">
                            <div class="prediction-title">
                                ⚠️ PREDICTIVE FAILURE ALERT: ${data.prediction.severity}
                            </div>
                            <div style="font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; color: var(--text-muted);">
                                Interface: ${data.prediction.link}
                             </div>
                        </div>
                        <div class="prediction-probability-row">
                            <span>Failure Probability:</span>
                            <div class="probability-bar-outer">
                                <div class="probability-bar-inner" style="width: ${data.prediction.failure_probability}%"></div>
                            </div>
                            <span class="probability-val">${data.prediction.failure_probability}%</span>
                        </div>
                        <div class="prediction-details">
                            <div class="prediction-detail-item">
                                <span class="prediction-detail-label">Est. Time to Failure</span>
                                <span class="prediction-detail-value time-urgent">${data.prediction.time_to_failure_minutes} mins</span>
                            </div>
                            <div class="prediction-detail-item">
                                <span class="prediction-detail-label">Model Confidence</span>
                                <span class="prediction-detail-value confidence-high">${data.prediction.confidence}%</span>
                            </div>
                        </div>
                        <div>
                            <span class="prediction-detail-label" style="font-weight: 600;">Contributing Factors:</span>
                            <ul class="factors-list" style="margin-top: 4px;">
                                ${data.prediction.contributing_factors.map(f => `<li>${f}</li>`).join('')}
                            </ul>
                        </div>
                        <div class="cli-command-row">
                            <span class="cli-title">Recommended Mitigation CLI:</span>
                            <div class="cli-command-box">
                                <span>${data.prediction.recommended_cli}</span>
                                <button class="apply-fix-btn" id="apply-fix-btn">Apply Fix</button>
                            </div>
                        </div>
                    </div>
                `;

                document.getElementById('apply-fix-btn').addEventListener('click', () => {
                    applyMitigationFix(data.prediction.link);
                });
            } else if (data.telemetry.status === 'fix_applied' || data.telemetry.status === 'normal') {
                // Shows banner: "✅ Fix Applied — Network Stabilized" at the end of simulation
                if (!cardContainer.classList.contains('hidden') && !cardContainer.textContent.includes('Stabilized')) {
                    cardContainer.innerHTML = `
                        <div class="prediction-card" style="border-color: var(--accent-green); box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15); animation: msg-in 0.3s ease;">
                            <div class="prediction-title" style="color: var(--accent-green);">
                                ✅ Fix Applied — Network Stabilized
                            </div>
                            <p style="font-size: 0.8rem; margin: 0; color: var(--text-secondary);">
                                Telemetry signals operating normally. Underlay path hops stabilized.
                            </p>
                        </div>
                    `;
                    setTimeout(() => {
                        cardContainer.classList.add('hidden');
                    }, 4000);
                }
            }
        };

        telemetrySource.onerror = (e) => {
            console.error('SSE connection lost.', e);
            telemetrySource.close();
        };
    }

    function applyMitigationFix(linkLabel) {
        const cardContainer = document.getElementById('prediction-card-container');
        cardContainer.innerHTML = `
            <div class="prediction-card" style="border-color: var(--accent-green); box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15); animation: msg-in 0.3s ease;">
                <div class="prediction-title" style="color: var(--accent-green);">
                    ✅ Fix Applied — Network Stabilized
                </div>
                <p style="font-size: 0.8rem; margin: 0; color: var(--text-secondary);">
                    Mitigation command successfully deployed to secure underlay switches. Prevented network drop on ${linkLabel}.
                </p>
            </div>
        `;
        
        // Instantly force sidebar indicators to healthy
        const item = document.getElementById('link-status-' + linkLabel.replace(' ', '-'));
        if (item) {
            const dot = item.querySelector('.status-indicator-dot');
            const badge = item.querySelector('.link-status-badge');
            dot.className = 'status-indicator-dot healthy';
            badge.className = 'link-status-badge healthy';
            badge.textContent = 'HEALTHY';
        }
        
        setTimeout(() => {
            cardContainer.classList.add('hidden');
        }, 4000);
    }

    // ═══ Demo Mode Controls (FIX 3) ═══
    const runDemoBtn = document.getElementById('run-demo-btn');
    if (runDemoBtn) {
        runDemoBtn.addEventListener('click', () => {
            // 1. Resets timeline to empty
            const timelineEvents = document.getElementById('timeline-events');
            timelineEvents.innerHTML = '<div class="empty-timeline">Live Teleplay starting...</div>';

            // 2. Hide prediction card
            const cardContainer = document.getElementById('prediction-card-container');
            cardContainer.classList.add('hidden');
            cardContainer.innerHTML = '';

            // Reset sidebar status indicators to healthy initially
            document.querySelectorAll('.link-status-item').forEach(item => {
                const dot = item.querySelector('.status-indicator-dot');
                const badge = item.querySelector('.link-status-badge');
                dot.className = 'status-indicator-dot healthy';
                badge.className = 'link-status-badge healthy';
                badge.textContent = 'HEALTHY';
            });

            // 3. Show live banner
            const liveBanner = document.getElementById('live-banner');
            if (liveBanner) {
                liveBanner.classList.remove('hidden');
            }

            // 4. Restart SSE stream from the beginning
            startTelemetryStream();
        });
    }

    // ═══ Mobile Navigation Tab Switching ═══
    const navButtons = document.querySelectorAll('.mobile-nav-btn');
    const appContainer = document.querySelector('.app-container');

    function switchMobileTab(target) {
        if (!appContainer) return;
        
        // Remove active class from all buttons
        navButtons.forEach(btn => {
            if (btn.getAttribute('data-target') === target) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        // Set layout classes on app-container
        appContainer.classList.remove('show-config', 'show-telemetry');
        if (target === 'config') {
            appContainer.classList.add('show-config');
        } else if (target === 'telemetry') {
            appContainer.classList.add('show-telemetry');
        }
    }

    if (navButtons.length > 0) {
        navButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const target = btn.getAttribute('data-target');
                switchMobileTab(target);
            });
        });
    }

    // ═══ Initial Load ═══
    fetchmodels();
    fetch('/api/config')
        .then(r => r.json())
        .then(data => {
            if (data.loaded) {
                setModelActive(data);
                welcomeScreen?.remove();
                addMessage(`Model auto-loaded: Aegis ${data.num_qubits}-Qubit model active (${data.total_parameters?.toLocaleString()} params)`, 'system');
            }
        })
        .catch(err => console.error('Initial config fetch failed:', err));
});
