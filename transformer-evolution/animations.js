// Animation Controller using GSAP
class AnimationController {
    constructor() {
        this.isAnimating = false;
        this.currentTimeline = null;

        // Register GSAP plugins
        gsap.registerPlugin(Flip);

        // Default easing
        this.defaultEase = "power2.inOut";
        this.springEase = "elastic.out(1, 0.5)";
    }

    // Main architecture transition animation
    async transitionArchitecture(oldArch, newArch, container, renderCallback) {
        if (this.isAnimating) return;
        this.isAnimating = true;

        const oldComponents = container.querySelectorAll('.arch-component');
        const oldConnections = container.querySelectorAll('.connection-line');

        // Get positions of old components
        const oldPositions = new Map();
        oldComponents.forEach(comp => {
            const rect = comp.getBoundingClientRect();
            oldPositions.set(comp.dataset.id, {
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height,
                element: comp
            });
        });

        // Find components that exist in both, only in old, only in new
        const oldIds = new Set(oldArch?.components.map(c => c.id) || []);
        const newIds = new Set(newArch.components.map(c => c.id));

        const persistingIds = [...oldIds].filter(id => newIds.has(id));
        const removedIds = [...oldIds].filter(id => !newIds.has(id));
        const addedIds = [...newIds].filter(id => !oldIds.has(id));

        // Create master timeline
        const tl = gsap.timeline({
            onComplete: () => {
                this.isAnimating = false;
            }
        });

        // Phase 1: Fade out removed components and connections
        if (removedIds.length > 0 || oldConnections.length > 0) {
            const removeElements = [];
            removedIds.forEach(id => {
                const el = container.querySelector(`[data-id="${id}"]`);
                if (el) removeElements.push(el);
            });

            tl.to([...removeElements, ...oldConnections], {
                opacity: 0,
                scale: 0.8,
                duration: 0.4,
                stagger: 0.02,
                ease: "power2.in"
            });
        }

        // Phase 2: Render new architecture (but hidden)
        tl.call(() => {
            renderCallback(newArch, true); // true = hidden initially
        });

        // Phase 3: Animate persisting components to new positions
        tl.call(() => {
            const newComponents = container.querySelectorAll('.arch-component');

            persistingIds.forEach(id => {
                const oldPos = oldPositions.get(id);
                const newComp = container.querySelector(`[data-id="${id}"]`);

                if (oldPos && newComp) {
                    const newRect = newComp.getBoundingClientRect();
                    const dx = oldPos.x - newRect.x;
                    const dy = oldPos.y - newRect.y;

                    // Start from old position
                    gsap.set(newComp, {
                        x: dx,
                        y: dy,
                        opacity: 1
                    });

                    // Animate to new position
                    gsap.to(newComp, {
                        x: 0,
                        y: 0,
                        duration: 0.6,
                        ease: this.defaultEase
                    });
                }
            });
        });

        // Phase 4: Fade in new components
        tl.call(() => {
            addedIds.forEach((id, index) => {
                const el = container.querySelector(`[data-id="${id}"]`);
                if (el) {
                    gsap.fromTo(el,
                        {
                            opacity: 0,
                            scale: 0.5,
                            y: 20
                        },
                        {
                            opacity: 1,
                            scale: 1,
                            y: 0,
                            duration: 0.5,
                            delay: index * 0.05,
                            ease: "back.out(1.7)"
                        }
                    );
                }
            });
        }, null, "+=0.3");

        // Phase 5: Animate connections
        tl.call(() => {
            const newConnections = container.querySelectorAll('.connection-line');
            newConnections.forEach((conn, index) => {
                const length = conn.getTotalLength ? conn.getTotalLength() : 100;
                gsap.set(conn, {
                    strokeDasharray: length,
                    strokeDashoffset: length,
                    opacity: 1
                });
                gsap.to(conn, {
                    strokeDashoffset: 0,
                    duration: 0.4,
                    delay: index * 0.03,
                    ease: "power1.inOut"
                });
            });
        }, null, "+=0.2");

        // Highlight changed components
        tl.call(() => {
            if (newArch.changes) {
                newArch.changes.forEach(change => {
                    // Find related components and pulse them
                    const relatedComps = this.findRelatedComponents(change, newArch);
                    relatedComps.forEach(comp => {
                        this.pulseComponent(container.querySelector(`[data-id="${comp}"]`));
                    });
                });
            }
        }, null, "+=0.3");

        return tl;
    }

    // Find components related to a change
    findRelatedComponents(change, arch) {
        const related = [];
        const title = change.title.toLowerCase();

        arch.components.forEach(comp => {
            const label = (comp.label + ' ' + (comp.sublabel || '')).toLowerCase();
            if (title.includes('attention') && comp.type === 'attention') related.push(comp.id);
            if (title.includes('norm') && comp.type === 'norm') related.push(comp.id);
            if (title.includes('ffn') && comp.type === 'ffn') related.push(comp.id);
            if (title.includes('embedding') && comp.type === 'embedding') related.push(comp.id);
            if (title.includes('rope') && label.includes('rope')) related.push(comp.id);
            if (title.includes('swiglu') && label.includes('swiglu')) related.push(comp.id);
            if (title.includes('gqa') && label.includes('gqa')) related.push(comp.id);
            if (title.includes('moe') && (label.includes('expert') || label.includes('router'))) related.push(comp.id);
        });

        return related;
    }

    // Pulse animation for highlighting
    pulseComponent(element) {
        if (!element) return;

        const bg = element.querySelector('.component-bg');
        if (bg) {
            gsap.to(bg, {
                filter: "brightness(1.5) drop-shadow(0 0 15px currentColor)",
                duration: 0.3,
                yoyo: true,
                repeat: 2,
                ease: "power1.inOut"
            });
        }
    }

    // Component hover animation
    componentHover(element, isEnter) {
        const bg = element.querySelector('.component-bg');
        if (!bg) return;

        if (isEnter) {
            gsap.to(element, {
                scale: 1.02,
                duration: 0.2,
                ease: "power2.out"
            });
            gsap.to(bg, {
                filter: "brightness(1.15)",
                duration: 0.2
            });
        } else {
            gsap.to(element, {
                scale: 1,
                duration: 0.2,
                ease: "power2.out"
            });
            gsap.to(bg, {
                filter: "brightness(1)",
                duration: 0.2
            });
        }
    }

    // Component selection animation
    selectComponent(element) {
        // Deselect animation for others
        document.querySelectorAll('.arch-component.selected').forEach(el => {
            if (el !== element) {
                gsap.to(el, {
                    scale: 1,
                    duration: 0.3,
                    ease: "power2.out"
                });
                el.classList.remove('selected');
            }
        });

        if (element) {
            element.classList.add('selected');
            gsap.fromTo(element,
                { scale: 1 },
                {
                    scale: 1.05,
                    duration: 0.3,
                    ease: "back.out(2)",
                    yoyo: true,
                    repeat: 1
                }
            );
        }
    }

    // Transition overlay animation
    showTransitionOverlay(title, subtitle) {
        const overlay = document.getElementById('transition-overlay');
        const titleEl = document.getElementById('transition-title');
        const descEl = document.getElementById('transition-description');
        const icon = overlay.querySelector('.transition-icon');
        const content = overlay.querySelector('.transition-content');

        titleEl.textContent = title;
        descEl.textContent = subtitle;

        const tl = gsap.timeline();

        tl.set(overlay, { visibility: 'visible' })
          .fromTo(overlay,
            { opacity: 0 },
            { opacity: 1, duration: 0.3 }
          )
          .fromTo(icon,
            { scale: 0, rotation: -180 },
            { scale: 1, rotation: 0, duration: 0.5, ease: "back.out(1.7)" },
            "-=0.1"
          )
          .fromTo(titleEl,
            { opacity: 0, y: 20 },
            { opacity: 1, y: 0, duration: 0.4 },
            "-=0.2"
          )
          .fromTo(descEl,
            { opacity: 0, y: 10 },
            { opacity: 1, y: 0, duration: 0.3 },
            "-=0.2"
          );

        return tl;
    }

    hideTransitionOverlay() {
        const overlay = document.getElementById('transition-overlay');

        return gsap.to(overlay, {
            opacity: 0,
            duration: 0.3,
            onComplete: () => {
                overlay.style.visibility = 'hidden';
            }
        });
    }

    // Panel slide animation
    slideInPanel(panel) {
        gsap.fromTo(panel,
            { x: 400, opacity: 0 },
            { x: 0, opacity: 1, duration: 0.4, ease: "power2.out" }
        );
    }

    slideOutPanel(panel) {
        gsap.to(panel, {
            x: 400,
            opacity: 0,
            duration: 0.3,
            ease: "power2.in"
        });
    }

    // Content fade animation for detail panel
    animateDetailContent() {
        const sections = document.querySelectorAll('#component-details .detail-section');

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

    // Timeline dot animation
    animateTimelineDot(dot, isActive) {
        if (isActive) {
            gsap.to(dot, {
                scale: 1.3,
                duration: 0.3,
                ease: "back.out(2)"
            });
        } else {
            gsap.to(dot, {
                scale: 1,
                duration: 0.2
            });
        }
    }

    // Connection line draw animation
    drawConnection(path, duration = 0.5) {
        const length = path.getTotalLength();

        gsap.set(path, {
            strokeDasharray: length,
            strokeDashoffset: length
        });

        gsap.to(path, {
            strokeDashoffset: 0,
            duration: duration,
            ease: "power1.inOut"
        });
    }

    // Stagger animation for multiple elements
    staggerIn(elements, options = {}) {
        const defaults = {
            opacity: 0,
            y: 20,
            duration: 0.4,
            stagger: 0.05,
            ease: "power2.out"
        };

        const settings = { ...defaults, ...options };

        gsap.fromTo(elements,
            { opacity: 0, y: settings.y },
            {
                opacity: 1,
                y: 0,
                duration: settings.duration,
                stagger: settings.stagger,
                ease: settings.ease
            }
        );
    }

    // Zoom animation
    animateZoom(container, targetScale, duration = 0.3) {
        gsap.to(container, {
            scale: targetScale,
            duration: duration,
            ease: "power2.out"
        });
    }

    // Shake animation for errors or invalid actions
    shake(element) {
        gsap.to(element, {
            x: [-5, 5, -5, 5, 0],
            duration: 0.4,
            ease: "power2.inOut"
        });
    }

    // Success animation
    success(element) {
        gsap.fromTo(element,
            { scale: 1 },
            {
                scale: 1.1,
                duration: 0.2,
                yoyo: true,
                repeat: 1,
                ease: "power2.out"
            }
        );
    }
}

// Global animation controller instance
window.animController = new AnimationController();
