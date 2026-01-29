// Main Application Logic
class TransformerExplorer {
    constructor() {
        this.currentArchitecture = 'original';
        this.previousArchitecture = null;
        this.selectedComponent = null;
        this.zoomLevel = 1;
        this.isDragging = false;
        this.dragStart = { x: 0, y: 0 };
        this.panOffset = { x: 0, y: 0 };
        this.isTransitioning = false;

        this.init();
    }

    init() {
        this.setupTimeline();
        this.setupEventListeners();
        this.renderArchitecture(this.currentArchitecture);
        this.loadTheme();
        this.updateSidebar(ARCHITECTURES[this.currentArchitecture]);

        // Initial animation
        gsap.from('.navbar', { y: -100, opacity: 0, duration: 0.6, ease: "power2.out" });
        gsap.from('.sidebar', { x: -100, opacity: 0, duration: 0.6, delay: 0.2, ease: "power2.out" });
        gsap.from('.detail-panel', { x: 100, opacity: 0, duration: 0.6, delay: 0.2, ease: "power2.out" });
    }

    // ==================== TIMELINE ====================
    setupTimeline() {
        const navTimeline = document.querySelector('.nav-timeline');
        const container = document.getElementById('timeline-items');
        container.innerHTML = '';

        // Define eras for grouping
        const eras = {
            'original': 'foundation',
            'gpt1': 'foundation',
            'bert': 'foundation',
            'gpt2': 'scaling',
            'llama': 'scaling',
            'llama2': 'scaling',
            'llama3': 'scaling',
            'rlhf': 'alignment',
            'dpo': 'alignment',
            'mixtral': 'frontier',
            'deepseek': 'frontier',
            'qwen2': 'frontier',
            'deepseekr1': 'frontier'
        };

        // Create SVG for curved path
        this.createTimelineSVG(navTimeline);

        // Add track labels
        this.addTrackLabels(navTimeline);

        ARCHITECTURE_ORDER.forEach((archId, index) => {
            const arch = ARCHITECTURES[archId];
            const item = document.createElement('div');
            item.className = 'timeline-item' + (index === 0 ? ' active' : '');
            item.dataset.arch = archId;
            item.dataset.category = arch.category;
            item.dataset.era = eras[archId] || 'foundation';

            // Add category indicator
            const categoryClass = arch.category === 'training' ? 'training' : 'arch';

            // Shorten names for space
            const shortNames = {
                'Original Transformer': 'Transformer',
                'LLaMA 3.1': 'LLaMA 3',
                'DeepSeek V2': 'DeepSeek',
                'Qwen 2.5': 'Qwen 2',
                'DeepSeek R1': 'R1'
            };
            const displayName = shortNames[arch.name] || arch.name;

            item.innerHTML = `
                <div class="timeline-dot ${categoryClass}"></div>
                <span class="timeline-label">${displayName}<br><small>${arch.year}</small></span>
            `;

            item.addEventListener('click', () => this.switchArchitecture(archId));

            // Hover animation
            item.addEventListener('mouseenter', () => {
                if (!item.classList.contains('active')) {
                    gsap.to(item.querySelector('.timeline-dot'), {
                        scale: 1.3,
                        duration: 0.2,
                        ease: "power2.out"
                    });
                }
            });
            item.addEventListener('mouseleave', () => {
                if (!item.classList.contains('active')) {
                    gsap.to(item.querySelector('.timeline-dot'), {
                        scale: 1,
                        duration: 0.2,
                        ease: "power2.out"
                    });
                }
            });

            container.appendChild(item);
        });

        this.updateTimelineProgress();
    }

    addTrackLabels(navTimeline) {
        // Remove existing labels
        navTimeline.querySelectorAll('.track-label').forEach(el => el.remove());

        // Architecture track label
        const archLabel = document.createElement('div');
        archLabel.className = 'track-label arch-track-label';
        archLabel.innerHTML = '<span>Architecture</span>';
        navTimeline.appendChild(archLabel);

        // Training track label
        const trainingLabel = document.createElement('div');
        trainingLabel.className = 'track-label training-track-label';
        trainingLabel.innerHTML = '<span>Training</span>';
        navTimeline.appendChild(trainingLabel);
    }

    createTimelineSVG(navTimeline) {
        // Remove existing SVG if present
        const existingSvg = navTimeline.querySelector('.timeline-svg');
        if (existingSvg) existingSvg.remove();

        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.classList.add('timeline-svg');
        svg.setAttribute('preserveAspectRatio', 'none');

        // Create gradients for progress
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');

        // Main architecture gradient (purple to pink)
        const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
        gradient.setAttribute('id', 'timeline-gradient');
        gradient.setAttribute('x1', '0%');
        gradient.setAttribute('y1', '0%');
        gradient.setAttribute('x2', '100%');
        gradient.setAttribute('y2', '0%');

        const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
        stop1.setAttribute('offset', '0%');
        stop1.setAttribute('style', 'stop-color:#6366f1');
        const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
        stop2.setAttribute('offset', '50%');
        stop2.setAttribute('style', 'stop-color:#8b5cf6');
        const stop3 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
        stop3.setAttribute('offset', '100%');
        stop3.setAttribute('style', 'stop-color:#ec4899');

        gradient.appendChild(stop1);
        gradient.appendChild(stop2);
        gradient.appendChild(stop3);
        defs.appendChild(gradient);

        // Training gradient (amber/orange)
        const trainingGradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
        trainingGradient.setAttribute('id', 'training-gradient');
        trainingGradient.setAttribute('x1', '0%');
        trainingGradient.setAttribute('y1', '0%');
        trainingGradient.setAttribute('x2', '100%');
        trainingGradient.setAttribute('y2', '0%');

        const tStop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
        tStop1.setAttribute('offset', '0%');
        tStop1.setAttribute('style', 'stop-color:#f59e0b');
        const tStop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
        tStop2.setAttribute('offset', '100%');
        tStop2.setAttribute('style', 'stop-color:#ef4444');

        trainingGradient.appendChild(tStop1);
        trainingGradient.appendChild(tStop2);
        defs.appendChild(trainingGradient);

        svg.appendChild(defs);

        // Main architecture track - background path
        const bgPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        bgPath.classList.add('timeline-path');
        bgPath.setAttribute('data-track', 'architecture');
        svg.appendChild(bgPath);

        // Main architecture track - progress path
        const progressPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        progressPath.classList.add('timeline-progress-path');
        progressPath.setAttribute('data-track', 'architecture');
        svg.appendChild(progressPath);

        // Training track - background path
        const trainingBgPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        trainingBgPath.classList.add('timeline-path', 'training-path');
        trainingBgPath.setAttribute('data-track', 'training');
        trainingBgPath.setAttribute('stroke', 'rgba(245, 158, 11, 0.3)');
        trainingBgPath.setAttribute('stroke-width', '2');
        svg.appendChild(trainingBgPath);

        // Training track - progress path
        const trainingProgressPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        trainingProgressPath.classList.add('timeline-progress-path', 'training-progress-path');
        trainingProgressPath.setAttribute('data-track', 'training');
        trainingProgressPath.setAttribute('stroke', 'url(#training-gradient)');
        trainingProgressPath.setAttribute('stroke-width', '2');
        svg.appendChild(trainingProgressPath);

        // Branch connectors (from main track to training track)
        const branchGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        branchGroup.classList.add('branch-connectors');
        svg.appendChild(branchGroup);

        // Insert before timeline-items
        const track = navTimeline.querySelector('.timeline-track');
        navTimeline.insertBefore(svg, track);

        // Update path after layout
        requestAnimationFrame(() => this.updateTimelinePath());
        window.addEventListener('resize', () => this.updateTimelinePath());
    }

    updateTimelinePath() {
        const svg = document.querySelector('.timeline-svg');
        if (!svg) return;

        const container = document.getElementById('timeline-items');
        const items = container.querySelectorAll('.timeline-item');
        if (items.length === 0) return;

        const rect = svg.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;

        svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

        // Separate architecture and training items
        const archPoints = [];
        const trainingPoints = [];
        const branchPoints = []; // Where training items branch from

        items.forEach((item) => {
            const dot = item.querySelector('.timeline-dot');
            const dotRect = dot.getBoundingClientRect();
            const svgRect = svg.getBoundingClientRect();

            const x = dotRect.left - svgRect.left + dotRect.width / 2;
            const y = dotRect.top - svgRect.top + dotRect.height / 2;
            const category = item.dataset.category;

            if (category === 'training') {
                trainingPoints.push({ x, y });
                // Calculate branch point (above the training dot on main track level)
                branchPoints.push({ x, y: height * 0.25, targetY: y });
            } else {
                archPoints.push({ x, y });
            }
        });

        // Create architecture track path
        const archPathD = this.createSmoothPath(archPoints);
        const bgPath = svg.querySelector('.timeline-path[data-track="architecture"]');
        const progressPath = svg.querySelector('.timeline-progress-path[data-track="architecture"]');

        if (bgPath) bgPath.setAttribute('d', archPathD);
        if (progressPath) {
            progressPath.setAttribute('d', archPathD);
            const length = progressPath.getTotalLength();
            progressPath.style.strokeDasharray = length;
        }

        // Create training track path
        if (trainingPoints.length > 0) {
            const trainingPathD = this.createSmoothPath(trainingPoints);
            const trainingBgPath = svg.querySelector('.timeline-path[data-track="training"]');
            const trainingProgressPath = svg.querySelector('.timeline-progress-path[data-track="training"]');

            if (trainingBgPath) trainingBgPath.setAttribute('d', trainingPathD);
            if (trainingProgressPath) {
                trainingProgressPath.setAttribute('d', trainingPathD);
                const length = trainingProgressPath.getTotalLength();
                trainingProgressPath.style.strokeDasharray = length;
            }

            // Create branch connectors
            const branchGroup = svg.querySelector('.branch-connectors');
            if (branchGroup) {
                branchGroup.innerHTML = '';
                branchPoints.forEach((bp) => {
                    // Vertical connector from main track to training track
                    const connector = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                    connector.setAttribute('d', `M ${bp.x} ${bp.y} Q ${bp.x} ${(bp.y + bp.targetY) / 2}, ${bp.x} ${bp.targetY - 8}`);
                    connector.setAttribute('stroke', 'rgba(245, 158, 11, 0.4)');
                    connector.setAttribute('stroke-width', '1.5');
                    connector.setAttribute('fill', 'none');
                    connector.setAttribute('stroke-dasharray', '4 3');
                    connector.classList.add('branch-connector');
                    branchGroup.appendChild(connector);
                });
            }
        }
    }

    createSmoothPath(points) {
        if (points.length < 2) return '';

        let d = `M ${points[0].x} ${points[0].y}`;

        for (let i = 1; i < points.length; i++) {
            const prev = points[i - 1];
            const curr = points[i];
            const next = points[i + 1];

            // Control points for smooth Bezier curves
            const tension = 0.3;
            let cp1x, cp1y, cp2x, cp2y;

            if (i === 1) {
                cp1x = prev.x + (curr.x - prev.x) * tension;
                cp1y = prev.y;
            } else {
                const prevPrev = points[i - 2];
                cp1x = prev.x + (curr.x - prevPrev.x) * tension;
                cp1y = prev.y + (curr.y - prevPrev.y) * tension;
            }

            if (!next) {
                cp2x = curr.x - (curr.x - prev.x) * tension;
                cp2y = curr.y;
            } else {
                cp2x = curr.x - (next.x - prev.x) * tension;
                cp2y = curr.y - (next.y - prev.y) * tension;
            }

            d += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${curr.x} ${curr.y}`;
        }

        return d;
    }

    updateTimelineProgress() {
        const currentIndex = ARCHITECTURE_ORDER.indexOf(this.currentArchitecture);
        const currentArch = ARCHITECTURES[this.currentArchitecture];
        const isTrainingItem = currentArch?.category === 'training';

        // Separate indices for each track
        const archItems = ARCHITECTURE_ORDER.filter(id => ARCHITECTURES[id]?.category !== 'training');
        const trainingItems = ARCHITECTURE_ORDER.filter(id => ARCHITECTURES[id]?.category === 'training');

        // Update architecture track progress
        const archProgressPath = document.querySelector('.timeline-progress-path[data-track="architecture"]');
        if (archProgressPath) {
            // Find how far along the architecture track we are
            let archProgress = 0;
            if (!isTrainingItem) {
                const archIndex = archItems.indexOf(this.currentArchitecture);
                archProgress = archIndex >= 0 ? archIndex / (archItems.length - 1) : 0;
            } else {
                // If on training item, find the nearest architecture item before it
                const currentOverallIndex = ARCHITECTURE_ORDER.indexOf(this.currentArchitecture);
                let lastArchIndex = -1;
                for (let i = currentOverallIndex; i >= 0; i--) {
                    if (ARCHITECTURES[ARCHITECTURE_ORDER[i]]?.category !== 'training') {
                        lastArchIndex = archItems.indexOf(ARCHITECTURE_ORDER[i]);
                        break;
                    }
                }
                archProgress = lastArchIndex >= 0 ? lastArchIndex / (archItems.length - 1) : 0;
            }

            const totalLength = archProgressPath.getTotalLength();
            const offset = totalLength * (1 - archProgress);
            gsap.to(archProgressPath, {
                strokeDashoffset: offset,
                duration: 0.6,
                ease: "power2.inOut"
            });
        }

        // Update training track progress
        const trainingProgressPath = document.querySelector('.timeline-progress-path[data-track="training"]');
        if (trainingProgressPath && trainingItems.length > 0) {
            let trainingProgress = 0;
            if (isTrainingItem) {
                const trainingIndex = trainingItems.indexOf(this.currentArchitecture);
                trainingProgress = trainingIndex >= 0 ? (trainingIndex + 1) / trainingItems.length : 0;
            } else {
                // Check if any training items come before current architecture
                const currentOverallIndex = ARCHITECTURE_ORDER.indexOf(this.currentArchitecture);
                let lastTrainingIndex = -1;
                for (let i = currentOverallIndex; i >= 0; i--) {
                    if (ARCHITECTURES[ARCHITECTURE_ORDER[i]]?.category === 'training') {
                        lastTrainingIndex = trainingItems.indexOf(ARCHITECTURE_ORDER[i]);
                        break;
                    }
                }
                trainingProgress = lastTrainingIndex >= 0 ? (lastTrainingIndex + 1) / trainingItems.length : 0;
            }

            const totalLength = trainingProgressPath.getTotalLength();
            const offset = totalLength * (1 - trainingProgress);
            gsap.to(trainingProgressPath, {
                strokeDashoffset: offset,
                duration: 0.6,
                ease: "power2.inOut"
            });
        }

        // Update dot states
        document.querySelectorAll('.timeline-item').forEach((item, index) => {
            const dot = item.querySelector('.timeline-dot');

            if (index < currentIndex) {
                item.classList.add('passed');
                item.classList.remove('active');
                gsap.to(dot, { scale: 1, duration: 0.2 });
            } else if (index === currentIndex) {
                item.classList.add('active');
                item.classList.remove('passed');
                gsap.to(dot, { scale: 1.2, duration: 0.3, ease: "back.out(2)" });
            } else {
                item.classList.remove('active', 'passed');
                gsap.to(dot, { scale: 1, duration: 0.2 });
            }
        });
    }

    // ==================== ARCHITECTURE SWITCHING ====================
    async switchArchitecture(archId) {
        if (archId === this.currentArchitecture || this.isTransitioning) return;

        this.isTransitioning = true;
        const oldArch = ARCHITECTURES[this.currentArchitecture];
        const newArch = ARCHITECTURES[archId];

        // Show transition overlay
        await this.showTransitionOverlay(newArch);

        // Store old positions for animation
        const oldComponents = document.querySelectorAll('.arch-component:not([data-container="true"])');
        const oldPositions = new Map();

        oldComponents.forEach(comp => {
            const rect = comp.getBoundingClientRect();
            const id = comp.dataset.id;
            oldPositions.set(id, {
                x: parseFloat(comp.getAttribute('transform')?.match(/translate\(([^,]+)/)?.[1] || 0),
                y: parseFloat(comp.getAttribute('transform')?.match(/,\s*([^)]+)/)?.[1] || 0),
                rect: rect
            });
        });

        // Update state
        this.previousArchitecture = this.currentArchitecture;
        this.currentArchitecture = archId;
        this.selectedComponent = null;

        // Update UI
        this.updateTimelineProgress();
        this.updateSidebar(newArch);
        this.resetDetailPanel();

        // Animate the transition
        await this.animateArchitectureTransition(oldArch, newArch, oldPositions);

        // Hide overlay
        await this.hideTransitionOverlay();

        this.isTransitioning = false;
    }

    async showTransitionOverlay(arch) {
        const overlay = document.getElementById('transition-overlay');
        const titleEl = document.getElementById('transition-title');
        const descEl = document.getElementById('transition-description');
        const icon = overlay.querySelector('.transition-icon');
        const content = overlay.querySelector('.transition-content');

        titleEl.textContent = arch.name;
        descEl.textContent = arch.keyInnovation;

        // Reset states
        gsap.set([icon, titleEl, descEl], { opacity: 0, y: 20 });
        gsap.set(icon, { scale: 0, rotation: -180 });

        return new Promise(resolve => {
            const tl = gsap.timeline({ onComplete: resolve });

            tl.to(overlay, { opacity: 1, visibility: 'visible', duration: 0.3 })
              .to(icon, {
                  scale: 1,
                  rotation: 0,
                  opacity: 1,
                  y: 0,
                  duration: 0.5,
                  ease: "back.out(1.7)"
              }, "-=0.1")
              .to(titleEl, {
                  opacity: 1,
                  y: 0,
                  duration: 0.4,
                  ease: "power2.out"
              }, "-=0.2")
              .to(descEl, {
                  opacity: 1,
                  y: 0,
                  duration: 0.3,
                  ease: "power2.out"
              }, "-=0.2");
        });
    }

    async hideTransitionOverlay() {
        const overlay = document.getElementById('transition-overlay');

        return new Promise(resolve => {
            gsap.to(overlay, {
                opacity: 0,
                duration: 0.4,
                ease: "power2.in",
                onComplete: () => {
                    overlay.style.visibility = 'hidden';
                    resolve();
                }
            });
        });
    }

    async animateArchitectureTransition(oldArch, newArch, oldPositions) {
        const container = document.getElementById('architecture-container');

        // Find common, removed, and added components
        const oldIds = new Set(oldArch?.components.map(c => c.id) || []);
        const newIds = new Set(newArch.components.map(c => c.id));

        const commonIds = [...oldIds].filter(id => newIds.has(id));
        const removedIds = [...oldIds].filter(id => !newIds.has(id));
        const addedIds = [...newIds].filter(id => !oldIds.has(id));

        // Phase 1: Fade out old architecture
        const oldSvg = container.querySelector('svg');
        if (oldSvg) {
            await new Promise(resolve => {
                gsap.to(oldSvg, {
                    opacity: 0,
                    scale: 0.95,
                    duration: 0.3,
                    ease: "power2.in",
                    onComplete: resolve
                });
            });
        }

        // Phase 2: Render new architecture (hidden)
        this.renderArchitecture(this.currentArchitecture, true);

        // Phase 3: Animate in new architecture
        const newSvg = container.querySelector('svg');
        const components = container.querySelectorAll('.arch-component');
        const connections = container.querySelectorAll('.connection-line');

        // Set initial states
        gsap.set(newSvg, { opacity: 0, scale: 0.95 });
        gsap.set(components, { opacity: 0 });
        gsap.set(connections, { opacity: 0 });

        // Animate SVG container
        await new Promise(resolve => {
            gsap.to(newSvg, {
                opacity: 1,
                scale: 1,
                duration: 0.4,
                ease: "power2.out",
                onComplete: resolve
            });
        });

        // Animate components with stagger
        const regularComponents = [...components].filter(c => !c.dataset.container);
        const containerComponents = [...components].filter(c => c.dataset.container);

        // Containers first
        gsap.to(containerComponents, {
            opacity: 1,
            duration: 0.3,
            stagger: 0.05
        });

        // Then regular components with different animations based on whether they're new
        regularComponents.forEach((comp, index) => {
            const id = comp.dataset.id;
            const isNew = addedIds.includes(id);
            const isCommon = commonIds.includes(id);

            if (isNew) {
                // New components: scale up from center
                gsap.fromTo(comp,
                    { opacity: 0, scale: 0.5, transformOrigin: 'center' },
                    {
                        opacity: 1,
                        scale: 1,
                        duration: 0.5,
                        delay: 0.1 + index * 0.03,
                        ease: "back.out(1.5)"
                    }
                );
            } else {
                // Existing components: fade in
                gsap.to(comp, {
                    opacity: 1,
                    duration: 0.4,
                    delay: index * 0.02,
                    ease: "power2.out"
                });
            }
        });

        // Animate connections (draw effect)
        connections.forEach((conn, index) => {
            const length = conn.getTotalLength ? conn.getTotalLength() : 200;

            gsap.set(conn, {
                strokeDasharray: length,
                strokeDashoffset: length,
                opacity: 1
            });

            gsap.to(conn, {
                strokeDashoffset: 0,
                duration: 0.5,
                delay: 0.3 + index * 0.02,
                ease: "power1.inOut"
            });
        });

        // Highlight changed components
        await new Promise(resolve => setTimeout(resolve, 600));

        if (newArch.changes && newArch.changes.length > 0) {
            this.highlightChangedComponents(newArch);
        }
    }

    highlightChangedComponents(arch) {
        const changedTypes = new Set();

        arch.changes.forEach(change => {
            const title = change.title.toLowerCase();
            if (title.includes('attention') || title.includes('gqa') || title.includes('mla') || title.includes('rope')) {
                changedTypes.add('attention');
            }
            if (title.includes('norm') || title.includes('rmsnorm') || title.includes('layernorm')) {
                changedTypes.add('norm');
            }
            if (title.includes('ffn') || title.includes('swiglu') || title.includes('expert') || title.includes('gelu')) {
                changedTypes.add('ffn');
            }
            if (title.includes('embedding') || title.includes('position')) {
                changedTypes.add('embedding');
                changedTypes.add('positional');
            }
        });

        document.querySelectorAll('.arch-component').forEach(comp => {
            const type = comp.dataset.type;
            if (changedTypes.has(type)) {
                const bg = comp.querySelector('.component-bg');
                if (bg) {
                    gsap.to(bg, {
                        filter: "brightness(1.4) drop-shadow(0 0 10px rgba(99, 102, 241, 0.6))",
                        duration: 0.3,
                        yoyo: true,
                        repeat: 2,
                        ease: "power1.inOut"
                    });
                }
            }
        });
    }

    updateSidebar(arch) {
        const titleEl = document.getElementById('arch-title');
        const yearEl = document.getElementById('arch-year');
        const descEl = document.getElementById('arch-description');
        const innovationEl = document.getElementById('key-innovation');

        // Animate text changes
        gsap.to([titleEl, yearEl, descEl, innovationEl], {
            opacity: 0,
            y: -10,
            duration: 0.2,
            onComplete: () => {
                titleEl.textContent = arch.name;
                yearEl.textContent = arch.year;
                descEl.textContent = arch.description;
                innovationEl.textContent = arch.keyInnovation;

                // Update paper link
                const metaItems = document.querySelectorAll('.meta-item');
                const paperLink = metaItems[0].querySelector('.meta-value');
                paperLink.href = arch.paperUrl;
                paperLink.textContent = arch.paper;

                gsap.to([titleEl, yearEl, descEl, innovationEl], {
                    opacity: 1,
                    y: 0,
                    duration: 0.3,
                    stagger: 0.05
                });
            }
        });

        // Update changes section
        const changesSection = document.getElementById('changes-section');
        const changesList = document.getElementById('changes-list');

        if (arch.changes && arch.changes.length > 0) {
            changesSection.style.display = 'block';

            gsap.to(changesList, {
                opacity: 0,
                duration: 0.2,
                onComplete: () => {
                    changesList.innerHTML = arch.changes.map(change => `
                        <div class="change-item">
                            <div class="change-icon ${change.type}">
                                ${this.getChangeIcon(change.type)}
                            </div>
                            <div class="change-content">
                                <h4>${change.title}</h4>
                                <p>${change.description}</p>
                            </div>
                        </div>
                    `).join('');

                    gsap.fromTo(changesList.children,
                        { opacity: 0, x: -20 },
                        { opacity: 1, x: 0, duration: 0.3, stagger: 0.1 }
                    );
                }
            });
        } else {
            gsap.to(changesSection, {
                opacity: 0,
                duration: 0.2,
                onComplete: () => {
                    changesSection.style.display = 'none';
                    changesSection.style.opacity = 1;
                }
            });
        }
    }

    getChangeIcon(type) {
        const icons = {
            added: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14M5 12h14"/></svg>',
            removed: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14"/></svg>',
            modified: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20V10M18 20V4M6 20v-4"/></svg>'
        };
        return icons[type] || icons.modified;
    }

    // ==================== ARCHITECTURE RENDERING ====================
    renderArchitecture(archId, hidden = false) {
        const arch = ARCHITECTURES[archId];
        const container = document.getElementById('architecture-container');

        // Calculate SVG dimensions
        const { width, height, offsetY } = this.calculateDimensions(arch);

        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('viewBox', `0 ${offsetY - 20} ${width} ${height + 40}`);
        svg.setAttribute('width', width);
        svg.setAttribute('height', height + 40);
        if (hidden) svg.style.opacity = '0';

        // Add defs for gradients and markers
        svg.innerHTML = this.getSVGDefs();

        // Render connections first (behind components)
        const connectionsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        connectionsGroup.classList.add('connections');
        arch.connections.forEach(conn => {
            const path = this.createConnectionPath(conn, arch.components);
            if (path) connectionsGroup.appendChild(path);
        });
        svg.appendChild(connectionsGroup);

        // Render components
        const componentsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        componentsGroup.classList.add('components');

        // Sort: containers first (behind), then regular components
        const sortedComponents = [...arch.components].sort((a, b) => {
            if (a.isContainer && !b.isContainer) return -1;
            if (!a.isContainer && b.isContainer) return 1;
            return 0;
        });

        sortedComponents.forEach(comp => {
            const element = this.createComponent(comp);
            componentsGroup.appendChild(element);
        });

        svg.appendChild(componentsGroup);

        // Add labels for encoder/decoder sections
        if (arch.id === 'original') {
            svg.appendChild(this.createSectionLabel('Encoder', 140, 580));
            svg.appendChild(this.createSectionLabel('Decoder', 380, 580));
        }

        // Replace old SVG
        container.innerHTML = '';
        container.appendChild(svg);

        // Reset zoom and pan
        this.zoomLevel = 1;
        this.panOffset = { x: 0, y: 0 };
        this.updateTransform();
    }

    calculateDimensions(arch) {
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;

        arch.components.forEach(comp => {
            minX = Math.min(minX, comp.x);
            maxX = Math.max(maxX, comp.x + comp.width);
            minY = Math.min(minY, comp.y);
            maxY = Math.max(maxY, comp.y + comp.height);
        });

        const padding = 60;
        return {
            width: maxX - minX + padding * 2,
            height: maxY - minY + padding * 2,
            offsetY: minY - padding
        };
    }

    getSVGDefs() {
        return `
            <defs>
                <!-- Gradients for components -->
                <linearGradient id="grad-embedding" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#3b82f6"/>
                    <stop offset="100%" style="stop-color:#2563eb"/>
                </linearGradient>
                <linearGradient id="grad-positional" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#06b6d4"/>
                    <stop offset="100%" style="stop-color:#0891b2"/>
                </linearGradient>
                <linearGradient id="grad-attention" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#8b5cf6"/>
                    <stop offset="100%" style="stop-color:#7c3aed"/>
                </linearGradient>
                <linearGradient id="grad-ffn" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#22c55e"/>
                    <stop offset="100%" style="stop-color:#16a34a"/>
                </linearGradient>
                <linearGradient id="grad-norm" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#f59e0b"/>
                    <stop offset="100%" style="stop-color:#d97706"/>
                </linearGradient>
                <linearGradient id="grad-output" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#ef4444"/>
                    <stop offset="100%" style="stop-color:#dc2626"/>
                </linearGradient>
                <linearGradient id="grad-softmax" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#f97316"/>
                    <stop offset="100%" style="stop-color:#ea580c"/>
                </linearGradient>

                <!-- Arrow marker -->
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#6366f1"/>
                </marker>
                <marker id="arrowhead-residual" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#ec4899"/>
                </marker>

                <!-- Glow filter -->
                <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                    <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
                    <feMerge>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>

                <!-- Selected glow -->
                <filter id="selected-glow" x="-50%" y="-50%" width="200%" height="200%">
                    <feGaussianBlur stdDeviation="6" result="coloredBlur"/>
                    <feFlood flood-color="#6366f1" flood-opacity="0.5"/>
                    <feComposite in2="coloredBlur" operator="in"/>
                    <feMerge>
                        <feMergeNode/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
            </defs>
        `;
    }

    createComponent(comp) {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.classList.add('arch-component');
        g.dataset.id = comp.id;
        g.dataset.type = comp.type;

        if (comp.isContainer) {
            g.dataset.container = 'true';

            // Container (dashed border)
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.classList.add('component-bg');
            rect.setAttribute('x', comp.x);
            rect.setAttribute('y', comp.y);
            rect.setAttribute('width', comp.width);
            rect.setAttribute('height', comp.height);
            rect.setAttribute('rx', '12');
            rect.setAttribute('fill', 'rgba(99, 102, 241, 0.03)');
            rect.setAttribute('stroke', '#6366f1');
            rect.setAttribute('stroke-width', '2');
            rect.setAttribute('stroke-dasharray', '8 4');
            rect.setAttribute('stroke-opacity', '0.5');
            g.appendChild(rect);

            // Container label (top-right)
            if (comp.sublabel) {
                const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                label.classList.add('component-sublabel');
                label.setAttribute('x', comp.x + comp.width - 10);
                label.setAttribute('y', comp.y + 20);
                label.setAttribute('text-anchor', 'end');
                label.setAttribute('fill', '#6366f1');
                label.setAttribute('font-weight', '600');
                label.setAttribute('font-size', '12');
                label.textContent = comp.sublabel;
                g.appendChild(label);
            }
        } else {
            // Regular component
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.classList.add('component-bg');
            rect.setAttribute('x', comp.x);
            rect.setAttribute('y', comp.y);
            rect.setAttribute('width', comp.width);
            rect.setAttribute('height', comp.height);
            rect.setAttribute('rx', '8');
            rect.setAttribute('fill', `url(#grad-${comp.type})`);
            rect.setAttribute('stroke', COMPONENT_COLORS[comp.type]?.stroke || '#666');
            rect.setAttribute('stroke-width', '2');
            g.appendChild(rect);

            // Component label
            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.classList.add('component-label');
            label.setAttribute('x', comp.x + comp.width / 2);
            label.setAttribute('y', comp.y + comp.height / 2 + (comp.sublabel ? -4 : 4));
            label.setAttribute('text-anchor', 'middle');
            label.textContent = comp.label;
            g.appendChild(label);

            // Sublabel if exists
            if (comp.sublabel) {
                const sublabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                sublabel.classList.add('component-sublabel');
                sublabel.setAttribute('x', comp.x + comp.width / 2);
                sublabel.setAttribute('y', comp.y + comp.height / 2 + 12);
                sublabel.setAttribute('text-anchor', 'middle');
                sublabel.textContent = comp.sublabel;
                g.appendChild(sublabel);
            }

            // Add click handler
            g.addEventListener('click', (e) => {
                e.stopPropagation();
                this.selectComponent(comp.id);
            });

            // Add hover effect with GSAP
            g.addEventListener('mouseenter', () => {
                gsap.to(g, {
                    scale: 1.03,
                    duration: 0.2,
                    ease: "power2.out",
                    transformOrigin: `${comp.x + comp.width/2}px ${comp.y + comp.height/2}px`
                });
                rect.setAttribute('filter', 'url(#glow)');
            });

            g.addEventListener('mouseleave', () => {
                if (!g.classList.contains('selected')) {
                    gsap.to(g, {
                        scale: 1,
                        duration: 0.2,
                        ease: "power2.out"
                    });
                    rect.removeAttribute('filter');
                }
            });
        }

        return g;
    }

    createConnectionPath(conn, components) {
        const fromComp = components.find(c => c.id === conn.from);
        const toComp = components.find(c => c.id === conn.to);

        if (!fromComp || !toComp || fromComp.isContainer || toComp.isContainer) return null;

        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');

        // Calculate connection points (center bottom to center top)
        const fromX = fromComp.x + fromComp.width / 2;
        const fromY = fromComp.y;
        const toX = toComp.x + toComp.width / 2;
        const toY = toComp.y + toComp.height;

        let d;
        const verticalDist = Math.abs(fromY - toY);
        const horizontalDist = Math.abs(fromX - toX);

        if (conn.type === 'cross') {
            // Cross-attention: smooth curve from encoder to decoder
            const midY = (fromY + toY) / 2;
            const ctrl1Y = fromY - verticalDist * 0.3;
            const ctrl2Y = toY + verticalDist * 0.3;
            d = `M ${fromX} ${fromY} C ${fromX} ${ctrl1Y}, ${toX} ${ctrl2Y}, ${toX} ${toY}`;
        } else if (conn.type === 'residual') {
            // Residual: side curve
            const offset = Math.min(35, horizontalDist * 0.3 + 20);
            d = `M ${fromX} ${fromY}
                 L ${fromX} ${fromY - 8}
                 Q ${fromX + offset} ${fromY - 8}, ${fromX + offset} ${(fromY + toY) / 2}
                 Q ${fromX + offset} ${toY + 8}, ${toX} ${toY + 8}
                 L ${toX} ${toY}`;
        } else {
            // Standard connection
            if (horizontalDist < 5) {
                // Straight line
                d = `M ${fromX} ${fromY} L ${toX} ${toY}`;
            } else {
                // Curved line
                const midY = (fromY + toY) / 2;
                d = `M ${fromX} ${fromY}
                     C ${fromX} ${midY}, ${toX} ${midY}, ${toX} ${toY}`;
            }
        }

        path.setAttribute('d', d);
        path.classList.add('connection-line');
        if (conn.type) path.classList.add(conn.type);

        if (conn.type === 'residual') {
            path.setAttribute('stroke', '#ec4899');
            path.setAttribute('stroke-dasharray', '6 4');
            path.setAttribute('marker-end', 'url(#arrowhead-residual)');
            path.setAttribute('opacity', '0.7');
        } else if (conn.type === 'cross') {
            path.setAttribute('stroke', '#6366f1');
            path.setAttribute('marker-end', 'url(#arrowhead)');
            path.setAttribute('opacity', '0.6');
        } else {
            path.setAttribute('stroke', '#4a4a5a');
            path.setAttribute('marker-end', 'url(#arrowhead)');
            path.setAttribute('opacity', '0.5');
        }

        path.setAttribute('stroke-width', '2');
        path.setAttribute('fill', 'none');

        return path;
    }

    createSectionLabel(text, x, y) {
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', x);
        label.setAttribute('y', y);
        label.setAttribute('text-anchor', 'middle');
        label.setAttribute('fill', '#6366f1');
        label.setAttribute('font-size', '14');
        label.setAttribute('font-weight', '600');
        label.setAttribute('opacity', '0.6');
        label.textContent = text;
        return label;
    }

    // ==================== COMPONENT SELECTION ====================
    selectComponent(componentId) {
        // Deselect previous
        document.querySelectorAll('.arch-component.selected').forEach(el => {
            el.classList.remove('selected');
            const bg = el.querySelector('.component-bg');
            if (bg) {
                bg.removeAttribute('filter');
                gsap.to(el, { scale: 1, duration: 0.2 });
            }
        });

        // Select new
        const element = document.querySelector(`[data-id="${componentId}"]`);
        if (element) {
            element.classList.add('selected');
            const bg = element.querySelector('.component-bg');
            if (bg) {
                bg.setAttribute('filter', 'url(#selected-glow)');
            }

            // Bounce animation
            gsap.fromTo(element,
                { scale: 1 },
                {
                    scale: 1.05,
                    duration: 0.15,
                    yoyo: true,
                    repeat: 1,
                    ease: "power2.inOut"
                }
            );
        }

        this.selectedComponent = componentId;
        this.showComponentDetails(componentId);

        // Open panel on mobile
        const panel = document.getElementById('detail-panel');
        panel.classList.add('open');
    }

    showComponentDetails(componentId) {
        const details = getComponentDetails(componentId);

        if (!details) {
            this.resetDetailPanel();
            return;
        }

        const placeholder = document.getElementById('placeholder-message');
        const detailsEl = document.getElementById('component-details');

        // Animate out placeholder
        gsap.to(placeholder, {
            opacity: 0,
            duration: 0.2,
            onComplete: () => {
                placeholder.style.display = 'none';
                detailsEl.style.display = 'block';

                // Update content
                document.getElementById('component-title').textContent = details.title;
                document.getElementById('component-overview').textContent = details.overview;
                document.getElementById('component-how').textContent = details.how;
                document.getElementById('component-math').textContent = details.math;

                const codeElement = document.getElementById('component-code');
                codeElement.textContent = details.code;
                hljs.highlightElement(codeElement);

                // Tips
                const tipsContainer = document.getElementById('component-tips');
                tipsContainer.innerHTML = details.tips.map(tip => `<li>${tip}</li>`).join('');

                // Animate in details
                const sections = detailsEl.querySelectorAll('.detail-section');
                gsap.fromTo(sections,
                    { opacity: 0, y: 15 },
                    {
                        opacity: 1,
                        y: 0,
                        duration: 0.4,
                        stagger: 0.08,
                        ease: "power2.out"
                    }
                );
            }
        });
    }

    resetDetailPanel() {
        const placeholder = document.getElementById('placeholder-message');
        const detailsEl = document.getElementById('component-details');

        gsap.to(detailsEl, {
            opacity: 0,
            duration: 0.2,
            onComplete: () => {
                detailsEl.style.display = 'none';
                placeholder.style.display = 'flex';
                gsap.to(placeholder, { opacity: 1, duration: 0.2 });
                document.getElementById('component-title').textContent = 'Select a Component';
            }
        });
    }

    // ==================== ZOOM & PAN ====================
    setupEventListeners() {
        // Zoom buttons
        document.getElementById('zoom-in').addEventListener('click', () => this.zoom(0.2));
        document.getElementById('zoom-out').addEventListener('click', () => this.zoom(-0.2));
        document.getElementById('zoom-reset').addEventListener('click', () => this.resetZoom());

        // Mouse wheel zoom
        const archView = document.querySelector('.architecture-view');
        archView.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? -0.1 : 0.1;
            this.zoom(delta);
        }, { passive: false });

        // Pan with mouse drag
        const container = document.getElementById('architecture-container');

        container.addEventListener('mousedown', (e) => {
            if (e.target.closest('.arch-component:not([data-container="true"])')) return;
            this.isDragging = true;
            this.dragStart = {
                x: e.clientX - this.panOffset.x,
                y: e.clientY - this.panOffset.y
            };
            container.style.cursor = 'grabbing';
        });

        document.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;
            this.panOffset = {
                x: e.clientX - this.dragStart.x,
                y: e.clientY - this.dragStart.y
            };
            this.updateTransform();
        });

        document.addEventListener('mouseup', () => {
            if (this.isDragging) {
                this.isDragging = false;
                container.style.cursor = 'grab';
            }
        });

        // Close detail panel
        document.getElementById('close-panel').addEventListener('click', () => {
            const panel = document.getElementById('detail-panel');
            panel.classList.remove('open');
            this.selectedComponent = null;

            document.querySelectorAll('.arch-component.selected').forEach(el => {
                el.classList.remove('selected');
                const bg = el.querySelector('.component-bg');
                if (bg) bg.removeAttribute('filter');
            });

            this.resetDetailPanel();
        });

        // Copy code button
        document.getElementById('copy-code').addEventListener('click', async () => {
            const code = document.getElementById('component-code').textContent;
            await navigator.clipboard.writeText(code);

            const btn = document.getElementById('copy-code');
            const originalHTML = btn.innerHTML;

            btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6L9 17l-5-5"/></svg><span>Copied!</span>';

            gsap.fromTo(btn, { scale: 0.95 }, { scale: 1, duration: 0.2, ease: "back.out(2)" });

            setTimeout(() => {
                btn.innerHTML = originalHTML;
            }, 2000);
        });

        // Theme toggle
        document.getElementById('theme-toggle').addEventListener('click', () => this.toggleTheme());

        // Click outside to deselect
        archView.addEventListener('click', (e) => {
            if (!e.target.closest('.arch-component:not([data-container="true"])') && !e.target.closest('.zoom-controls')) {
                this.selectedComponent = null;
                document.querySelectorAll('.arch-component.selected').forEach(el => {
                    el.classList.remove('selected');
                    const bg = el.querySelector('.component-bg');
                    if (bg) bg.removeAttribute('filter');
                    gsap.to(el, { scale: 1, duration: 0.2 });
                });
            }
        });

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                document.getElementById('detail-panel').classList.remove('open');
            }
            if (e.key === 'ArrowRight' && !this.isTransitioning) {
                this.navigateArchitecture(1);
            }
            if (e.key === 'ArrowLeft' && !this.isTransitioning) {
                this.navigateArchitecture(-1);
            }
        });

        // Touch support for mobile
        let touchStartX = 0;
        archView.addEventListener('touchstart', (e) => {
            touchStartX = e.touches[0].clientX;
        }, { passive: true });

        archView.addEventListener('touchend', (e) => {
            const touchEndX = e.changedTouches[0].clientX;
            const diff = touchStartX - touchEndX;

            if (Math.abs(diff) > 50 && !this.isTransitioning) {
                if (diff > 0) {
                    this.navigateArchitecture(1);
                } else {
                    this.navigateArchitecture(-1);
                }
            }
        }, { passive: true });
    }

    navigateArchitecture(direction) {
        const currentIndex = ARCHITECTURE_ORDER.indexOf(this.currentArchitecture);
        const newIndex = currentIndex + direction;

        if (newIndex >= 0 && newIndex < ARCHITECTURE_ORDER.length) {
            this.switchArchitecture(ARCHITECTURE_ORDER[newIndex]);
        }
    }

    zoom(delta) {
        const newZoom = Math.max(0.5, Math.min(2.5, this.zoomLevel + delta));

        gsap.to(this, {
            zoomLevel: newZoom,
            duration: 0.3,
            ease: "power2.out",
            onUpdate: () => this.updateTransform()
        });
    }

    resetZoom() {
        gsap.to(this, {
            zoomLevel: 1,
            duration: 0.4,
            ease: "power2.out",
            onUpdate: () => this.updateTransform()
        });

        gsap.to(this.panOffset, {
            x: 0,
            y: 0,
            duration: 0.4,
            ease: "power2.out",
            onUpdate: () => this.updateTransform()
        });
    }

    updateTransform() {
        const container = document.getElementById('architecture-container');
        container.style.transform = `translate(${this.panOffset.x}px, ${this.panOffset.y}px) scale(${this.zoomLevel})`;
    }

    // ==================== THEME ====================
    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';

        // Animate transition
        gsap.to('body', {
            opacity: 0.8,
            duration: 0.15,
            onComplete: () => {
                document.documentElement.setAttribute('data-theme', newTheme);
                localStorage.setItem('theme', newTheme);
                gsap.to('body', { opacity: 1, duration: 0.15 });
            }
        });
    }

    loadTheme() {
        const savedTheme = localStorage.getItem('theme') || 'dark';
        document.documentElement.setAttribute('data-theme', savedTheme);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new TransformerExplorer();
});
