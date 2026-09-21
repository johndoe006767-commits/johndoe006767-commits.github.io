import * as T from 'three';
// Centered organ meshes, keyed by tower id, reused by the activity-flow endpoint models.
export const organGeometries = new Map<string, T.BufferGeometry>();
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { TOWERS, type AnatomyController } from './tower-organ-mapping';
import { buildVesselNetwork } from './anatomy-vessels';

type Part = {
  group: string;
  id: string;
  name: string;
  positions: number;
  normals: number;
  indices: number;
  vertexCount: number;
  indexCount: number;
  bounds: [number[], number[]];
};
export type TowerOverviewController = AnatomyController & {
  setView: (yaw: number, pitch: number, zoom: number, journey: number) => void;
  // Builds (or rebuilds) one glowing neuron per process on the organ's surface.
  setProcessCount: (organId: string, count: number) => void;
};
type Callbacks = {
  onNodePosition: (
    index: number,
    x: number,
    y: number,
    opacity: number,
  ) => void;
  onOrganSelect: (id: string) => void;
  onReady: () => void;
  onProgress: (message: string) => void;
  onError: (message: string) => void;
};
type Organ = {
  id: string;
  mesh: T.Mesh<T.BufferGeometry, T.MeshPhysicalMaterial>;
  center: T.Vector3;
  size: T.Vector3;
  expandedScale: number;
  label: HTMLElement | null;
  side: number;
  row: number;
  hover: number;
  basePosition: T.Vector3;
  baseScale: number;
};
type Vessel = {
  mesh: T.Mesh<T.BufferGeometry, T.MeshPhysicalMaterial>;
  organ: Organ;
  uniforms: {
    uSeparation: { value: number };
    uTime: { value: number };
    uHoverOffset: { value: T.Vector3 };
  };
};

// The anatomy stays in Three.js. Continuous input never triggers React rendering.
export function createAnatomyScene(
  host: HTMLElement,
  labelHost: HTMLElement,
  callbacks: Callbacks,
): TowerOverviewController {
  const abort = new AbortController();
  let disposed = false,
    frame = 0,
    target = 0,
    progress = 0,
    paused = false,
    ready = false,
    time = 0,
    lastTime = 0,
    lastProgress = -1;
  let width = 1,
    height = 1,
    mobile = false;
  const preference = matchMedia('(prefers-reduced-motion: reduce)');
  let reduced = preference.matches;
  const onPreference = () => {
    reduced = preference.matches;
  };
  preference.addEventListener('change', onPreference);
  let renderer: T.WebGLRenderer;
  try {
    renderer = new T.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
  } catch {
    callbacks.onError(
      'Your browser could not start the 3D anatomy. Enable WebGL or try another browser.',
    );
    return {
      setView: () => {},
      setProcessCount: () => {},
      focusOrgan: () => {},
      setProgress: () => {},
      setPaused: () => {},
      destroy: () => {
        preference.removeEventListener('change', onPreference);
      },
    };
  }
  renderer.setClearColor('#f2f2f0', 0);
  renderer.outputColorSpace = T.SRGBColorSpace;
  renderer.toneMapping = T.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.94;
  host.appendChild(renderer.domElement);
  renderer.domElement.setAttribute('role', 'img');
  renderer.domElement.setAttribute(
    'aria-label',
    'A human skeleton with a beating GFS KL heart and eight connected organs. Use the horizontal slider to separate and reassemble the organs.',
  );
  const scene = new T.Scene();
  const camera = new T.OrthographicCamera(-2, 2, 1.9, -0.1, 0.01, 20);
  camera.position.set(0, 0.95, 5);
  camera.lookAt(0, 0.95, 0);
  scene.add(new T.HemisphereLight(0xfffcf5, 0x92968a, 0.85));
  const key = new T.DirectionalLight(0xfff8ee, 2.8);
  key.position.set(-2, 3, 4);
  scene.add(key);
  const fill = new T.DirectionalLight(0xe7ebed, 1.2);
  fill.position.set(2, 1, -2);
  scene.add(fill);
  const pmrem = new T.PMREMGenerator(renderer);
  const room = new RoomEnvironment();
  const environment = pmrem.fromScene(room, 0.06);
  scene.environment = environment.texture;
  pmrem.dispose();
  room.dispose();
  const ownedGeometries = new Set<T.BufferGeometry>(),
    ownedMaterials = new Set<T.Material>();
  const organs: Organ[] = [];
  const vessels: Vessel[] = [];
  let heart: Organ | undefined;
  const boneMaterial = new T.MeshPhysicalMaterial({
    color: '#c6c2b1',
    roughness: 0.48,
    metalness: 0.025,
    envMapIntensity: 0.25,
    clearcoat: 0.12,
  });
  ownedMaterials.add(boneMaterial);
  const skullMaterial = boneMaterial.clone();
  skullMaterial.transparent = true;
  skullMaterial.opacity = 0.42;
  skullMaterial.depthWrite = false;
  ownedMaterials.add(skullMaterial);
  const labels = [...labelHost.querySelectorAll<HTMLElement>('[data-organ]')];
  const heartWord = labelHost.querySelector<HTMLElement>(
    '[data-organ="heart"] > span',
  );
  const temp = new T.Vector3();
  let focusedOrgan = 'brain';
  let yaw = 0,
    pitch = 0,
    zoom = 1,
    journey = 0;
  // One neuron set per organ, built lazily once the tower's process count is known.
  const neuronSets = new Map<string, T.Group[]>();
  const activeNeurons = () => neuronSets.get(focusedOrgan) ?? [];
  const triggers = new Map(
    [...labelHost.querySelectorAll<HTMLElement>('[data-organ-trigger]')].map(
      (el) => [el.dataset.organTrigger!, el],
    ),
  );
  const activeTrigger = () => triggers.get(focusedOrgan);
  let anchor: HTMLElement | null = null,
    flyRenderer: T.WebGLRenderer | undefined,
    flyMesh: T.Mesh | undefined,
    flyEnvironment: T.WebGLRenderTarget | undefined;
  const flyScene = new T.Scene(),
    flyCamera = new T.OrthographicCamera(0, 1, 1, 0, 0.1, 1000);
  flyCamera.position.z = 500;

  const flyKey = key.clone(),
    flyFill = fill.clone();
  flyKey.position.set(-200, 300, 400);
  flyFill.position.set(200, 100, 300);
  flyScene.add(
    new T.HemisphereLight(0xfffcf5, 0x92968a, 0.85),
    flyKey,
    flyFill,
  );
  let flight = 0,
    flightStart = 0,
    returning = false;
  const flightFrom = new T.Vector3();
  let flightScale = 1;
  function brainScreen() {
    const brain = organs.find((o) => o.id === focusedOrgan)!;
    const rect = host.getBoundingClientRect();
    const point = brain.mesh.position.clone().project(camera);
    return {
      brain,
      x: rect.left + ((point.x + 1) * width) / 2,
      y: rect.top + ((1 - point.y) * height) / 2,
      scale: (brain.mesh.scale.x * height) / (camera.top - camera.bottom),
    };
  }
  function disposeNeurons(nodes: T.Group[]) {
    for (const node of nodes) {
      node.removeFromParent();
      node.traverse((child) => {
        if (!(child instanceof T.Mesh)) return;
        ownedGeometries.delete(child.geometry);
        child.geometry.dispose();
        const material = child.material as T.Material;
        ownedMaterials.delete(material);
        material.dispose();
      });
    }
  }
  function prepareNeurons(brain:Organ, nodeCount:number){
        const existing = neuronSets.get(brain.id);
        if (existing && existing.length === nodeCount) return;
        if (existing) disposeNeurons(existing);
        const neurons: T.Group[] = [];
        neuronSets.set(brain.id, neurons);
        const size = brain.size;
        // Project every dendrite onto the actual organ surface, rather than a flat plane.
        const source = brain.mesh.geometry;
        const bounds = source.boundingBox!,
          span = bounds.getSize(new T.Vector3());
        const positions = source.getAttribute('position'),
          indices = source.getIndex()!;
        const buckets = new Map<number, number[]>();
        for (let index = 0; index < indices.count; index += 3) {
          const vertex = indices.getX(index);
          const x = Math.min(
            3,
            Math.floor(((positions.getX(vertex) - bounds.min.x) / span.x) * 4),
          );
          const y = Math.min(
            3,
            Math.floor(((positions.getY(vertex) - bounds.min.y) / span.y) * 4),
          );
          const z = Math.min(
            3,
            Math.floor(((positions.getZ(vertex) - bounds.min.z) / span.z) * 4),
          );
          const key = x + y * 4 + z * 16,
            bucket = buckets.get(key) ?? [];
          bucket.push(vertex, indices.getX(index + 1), indices.getX(index + 2));
          buckets.set(key, bucket);
        }
        // Spatial partitions retain every triangle but reject most triangles before ray tests.
        const surfaces = [...buckets.values()].map((index) => {
          const geometry = new T.BufferGeometry();
          geometry.setAttribute('position', positions);
          geometry.setIndex(index);
          const box = new T.Box3();
          for (const vertex of index)
            box.expandByPoint(
              new T.Vector3().fromBufferAttribute(positions, vertex),
            );
          geometry.boundingBox = box;
          geometry.boundingSphere = box.getBoundingSphere(new T.Sphere());
          const mesh = new T.Mesh(geometry, brain.mesh.material);
          mesh.updateMatrixWorld(true);
          return mesh;
        });
        const surfaceRay = new T.Raycaster();
        const radius = size.length();
        const surface = (direction: T.Vector3) => {
          const d = direction.clone().normalize();
          surfaceRay.set(d.clone().multiplyScalar(radius), d.clone().negate());
          const hit = surfaceRay.intersectObjects(surfaces, false)[0];
          return hit
            ? hit.point.clone().addScaledVector(d, size.x * 0.002)
            : d.multiplyScalar(size.x * 0.42);
        };
        for (let i = 0; i < nodeCount; i++) {
          const angle = i * 2.399963229728653;
          const h = 0.62 - (1.24 * (i + 0.5)) / nodeCount;
          const direction = new T.Vector3(
            Math.sin(angle) * Math.sqrt(1 - h * h),
            h,
            Math.cos(angle) * Math.sqrt(1 - h * h),
          );
          const center = surface(direction);
          const projected = new Map<string, T.Vector3>();
          const tangent = new T.Vector3()
            .crossVectors(direction, new T.Vector3(0, 1, 0))
            .normalize();
          const vertical = new T.Vector3()
            .crossVectors(tangent, direction)
            .normalize();
          const group = new T.Group();
          group.position.copy(center);
          group.userData.normal = direction;
          const mat = new T.MeshPhysicalMaterial({
            color: '#c98b0b',
            emissive: '#eaa51b',
            emissiveIntensity: 0.48,
            metalness: 0.45,
            roughness: 0.26,
            clearcoat: 0.2,
            transparent: true,
            depthTest: true,
            depthWrite: false,
          });
          ownedMaterials.add(mat);
          // A soft halo stays within the dendrite footprint; anatomy size is unchanged.
          const haloMaterial = new T.ShaderMaterial({
            transparent: true, depthWrite: false, depthTest: true,
            blending: T.AdditiveBlending,
            uniforms: { strength: { value: 0 } },
            vertexShader: 'varying vec2 vUv; void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
            fragmentShader: 'varying vec2 vUv; uniform float strength; void main(){float r=length(vUv-.5)*2.;float a=exp(-r*r*5.)*(1.-smoothstep(.55,1.,r));gl_FragColor=vec4(1.,.58,.045,a*strength);}',
          });
          const haloGeometry = new T.PlaneGeometry(size.x * 0.15, size.x * 0.15);
          ownedMaterials.add(haloMaterial);
          ownedGeometries.add(haloGeometry);
          const halo = new T.Mesh(haloGeometry, haloMaterial);
          halo.quaternion.setFromUnitVectors(new T.Vector3(0,0,1), direction);
          halo.position.copy(direction).multiplyScalar(size.x * 0.009);
          group.add(halo);
          const sphere = new T.SphereGeometry(size.x * 0.014, 12, 10);
          ownedGeometries.add(sphere);
          group.add(new T.Mesh(sphere, mat));
          function project(u: number, v: number) {
            const key = u.toFixed(7) + ',' + v.toFixed(7);
            if (!projected.has(key))
              projected.set(
                key,
                surface(
                  center
                    .clone()
                    .addScaledVector(tangent, u)
                    .addScaledVector(vertical, v),
                ).sub(center),
              );
            return projected.get(key)!.clone();
          }
          function dendrite(points: T.Vector3[], thickness: number) {
            const g = new T.TubeGeometry(
              new T.CatmullRomCurve3(points),
              16,
              thickness,
              5,
              false,
            );
            ownedGeometries.add(g);
            group.add(new T.Mesh(g, mat));
          }
          for (let branch = 0; branch < 6; branch++) {
            const angle = (branch * Math.PI) / 3 + i * 0.7,
              len = size.x * (0.048 + (branch % 2) * 0.018);
            const u = Math.cos(angle) * len,
              v = Math.sin(angle) * len;
            dendrite(
              [
                project(0, 0),
                project(u * 0.32, v * 0.2),
                project(u * 0.68, v * 0.74),
                project(u, v),
              ],
              size.x * 0.0028,
            );
            for (const sign of [-1, 1]) {
              const eu = u + Math.cos(angle + sign * 0.7) * len * 0.4,
                ev = v + Math.sin(angle + sign * 0.7) * len * 0.4;
              dendrite(
                [project(u * 0.72, v * 0.72), project(u, v), project(eu, ev)],
                size.x * 0.0016,
              );
            }
          }
            neurons.push(group);
        }
        surfaces.forEach((mesh) => mesh.geometry.dispose());
  }
  function focusOrgan(organId: string, next: HTMLElement | null) {
    if (next && focusedOrgan !== organId && flyRenderer) {
      const previous = organs.find((o) => o.id === focusedOrgan);
      if (previous) previous.mesh.visible = true;
      const previousTrigger = activeTrigger();
      if (previousTrigger) previousTrigger.style.visibility = 'visible';
      if (flyMesh) flyScene.remove(flyMesh);
      flyMesh = undefined;
      flyEnvironment?.dispose();
      flyEnvironment = undefined;
      flyRenderer.dispose();
      flyRenderer.forceContextLoss();
      flyRenderer.domElement.remove();
      flyRenderer = undefined;
      flyCamera.right = 0;
      flight = 0;
      anchor = null;
    }
    if (next) focusedOrgan = organId;
    if (anchor === next || !ready) return;
    anchor = next;
    const start = brainScreen();
    if (!flyRenderer && next) {
      flyRenderer = new T.WebGLRenderer({ alpha: true, antialias: true });
      const generator = new T.PMREMGenerator(flyRenderer);
      const room = new RoomEnvironment();
      flyEnvironment = generator.fromScene(room, 0.06);
      flyScene.environment = flyEnvironment.texture;
      generator.dispose();
      room.dispose();
      flyRenderer.setClearColor(0, 0);
      flyRenderer.outputColorSpace = renderer.outputColorSpace;
      flyRenderer.toneMapping = renderer.toneMapping;
      flyRenderer.toneMappingExposure = renderer.toneMappingExposure;
      flyRenderer.domElement.className = 'flying-brain testing-flight';
      flyRenderer.domElement.setAttribute('aria-hidden', 'true');
      (host.closest('.testing') ?? document.body).appendChild(
        flyRenderer.domElement,
      );
      flyMesh = new T.Mesh(
        start.brain.mesh.geometry,
        start.brain.mesh.material,
      );
      flyMesh.position.set(start.x, innerHeight - start.y, 0);
      flyMesh.scale.setScalar(start.scale);
      flyScene.add(flyMesh);
      activeNeurons().forEach((node) => flyMesh!.add(node));
    }
    if (!flyMesh) return;
    start.brain.mesh.visible = false;
    clearHover();
    flightFrom.copy(flyMesh.position);
    flightScale = flyMesh.scale.x;
    flightStart = performance.now();
    flight = 1;
    returning = !next;
    const trigger = activeTrigger();
    if (trigger) trigger.style.visibility = 'hidden';
  }
  let visualJourney = 0,
    visualZoom = 1,
    lastFlightTime = 0;
  function renderFlight(now: number) {
    const delta = Math.min(0.05, (now - lastFlightTime) / 1000);
    lastFlightTime = now;
    // Soft, continuous follow of the wheel so neurons glide past rather than step.
    visualJourney = T.MathUtils.damp(visualJourney, journey, reduced ? 20 : 7, delta);
    visualZoom = T.MathUtils.damp(visualZoom, zoom, reduced ? 20 : 9, delta);
    if (!flyRenderer || !flyMesh) return;
    // Portal teardown is also a return signal, including interrupted close transitions.
    if (anchor && !anchor.isConnected) focusOrgan(focusedOrgan, null);
    const start = brainScreen();
    const rect = anchor?.getBoundingClientRect();
    const x = rect ? rect.left + rect.width / 2 : start.x;
    const y = rect ? rect.top + rect.height / 2 : start.y;
    const scale = rect
      ? Math.min(
          rect.width / start.brain.size.x,
          rect.height / start.brain.size.y,
        ) * 0.83
      : start.scale;
    const t =
      document.documentElement.dataset.input === 'keyboard'
        ? 1
        : Math.min(1, (now - flightStart) / (reduced ? 360 : 680));
    // On-screen travel from the body to centre stage: strong ease-in-out.
    const eased = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    const neurons = activeNeurons(),
      nodeCount = neurons.length;
    flyMesh.position
      .copy(flightFrom)
      .lerp(new T.Vector3(x, innerHeight - y, 0), eased);
    flyMesh.scale.setScalar(
      T.MathUtils.lerp(
        flightScale,
        scale * (returning ? 1 : visualZoom),
        eased,
      ),
    );
    flyMesh.rotation.set(
      returning ? 0 : pitch,
      returning
        ? start.brain.mesh.rotation.y * eased
        : yaw - visualJourney * (nodeCount - 1) * 2.399963229728653,
      0,
    );
    flyMesh.updateMatrixWorld(true);
    neurons.forEach((node, i) => {
      const distance = Math.abs(i - visualJourney * (nodeCount - 1));
      const targetOpacity = returning
        ? 0
        : T.MathUtils.smoothstep(visualJourney, 0.002, 0.045) * (1 - T.MathUtils.smoothstep(distance, 0.35, 1.65));
      const opacity = reduced ? targetOpacity : T.MathUtils.damp(node.userData.opacity ?? 0, targetOpacity, 7, delta);
      node.userData.opacity = opacity;
      node.visible = opacity > 0.002;
      const pulse = reduced ? 1 : 0.5 + 0.5 * Math.sin(now * 0.003 + i * 0.8);
      node.traverse((child) => {
        if (child instanceof T.Mesh && child.material instanceof T.ShaderMaterial) {
          child.material.uniforms.strength.value = opacity * (reduced ? 0.5 : 0.5 + pulse * 0.25);
        } else if (child instanceof T.Mesh) {
          const material = child.material as T.MeshPhysicalMaterial;
          material.opacity = opacity;
          material.emissiveIntensity = reduced ? 0.48 : 0.38 + pulse * 0.3;
        }
      });
      const point = node.getWorldPosition(new T.Vector3());
      const facing = (node.userData.normal as T.Vector3)
        .clone()
        .transformDirection(flyMesh!.matrixWorld).z;
      callbacks.onNodePosition(
        i,
        point.x,
        innerHeight - point.y,
        opacity * T.MathUtils.smoothstep(facing, 0.08, 0.38),
      );
    });
    if (flyCamera.right !== innerWidth || flyCamera.top !== innerHeight) {
      flyRenderer.setPixelRatio(Math.min(devicePixelRatio, 2));
      flyRenderer.setSize(innerWidth, innerHeight);
      flyCamera.right = innerWidth;
      flyCamera.top = innerHeight;
      flyCamera.updateProjectionMatrix();
    }
    flyRenderer.render(flyScene, flyCamera);
    if (t === 1 && returning) {
      start.brain.mesh.visible = true;
      flyScene.remove(flyMesh);
      flyMesh = undefined;
      flyEnvironment?.dispose();
      flyEnvironment = undefined;
      flyRenderer.dispose();
      flyRenderer.forceContextLoss();
      flyRenderer.domElement.remove();
      flyRenderer = undefined;
      flyCamera.right = 0;
      flight = 0;
      const trigger = activeTrigger();
      if (trigger) {
        trigger.style.visibility = 'visible';
        trigger.focus({ preventScroll: true });
      }
    }
  }
  let hovered: string | null = null;
  const raycaster = new T.Raycaster(),
    pointer = new T.Vector2();
  const onPointer = (event: PointerEvent) => {
    if (!ready || anchor || flight) return;
    const rect = host.getBoundingClientRect();
    pointer.set(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      1 - ((event.clientY - rect.top) / rect.height) * 2,
    );
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(
      organs.map((o) => o.mesh),
      false,
    )[0];
    const next = hit ? String(hit.object.userData.organ) : null;
    if (next === hovered) return;
    hovered = next;
    for (const organ of organs) {
      organ.mesh.material.emissive.set(
        organ.id === next ? '#4b1437' : '#000000',
      );
      organ.mesh.material.emissiveIntensity = organ.id === next ? 0.12 : 0;
      organ.label?.classList.toggle('emphasized', organ.id === next);
    }
    // Future navigation attaches to mesh.userData.towerId, not a screen coordinate.
  };
  const clearHover = () => {
    hovered = null;
    for (const o of organs) {
      o.mesh.material.emissiveIntensity = 0;
      o.label?.classList.remove('emphasized');
    }
  };
  const onClick = (event: PointerEvent) => {
    onPointer(event);
    // Every organ is a tower; the heart is not in `organs`, so it never selects.
    if (hovered && !anchor && !flight) callbacks.onOrganSelect(hovered);
  };
  host.addEventListener('pointerup', onClick);
  host.addEventListener('pointermove', onPointer);
  host.addEventListener('pointerleave', clearHover);
  const resize = () => {
    if (
      disposed ||
      !host.isConnected ||
      host.clientWidth < 1 ||
      host.clientHeight < 1
    )
      return;
    width = host.clientWidth;
    height = host.clientHeight;
    mobile = width < 768;
    renderer.setPixelRatio(Math.min(devicePixelRatio, mobile ? 1.5 : 2));
    renderer.setSize(width, height);
    // Frame the whole figure, head to feet, beneath the site navigation bar.
    const span = mobile ? 2.2 : 2.06;
    camera.left = (-span * width) / height / 2;
    camera.right = -camera.left;
    camera.top = span / 2;
    camera.bottom = -span / 2;
    camera.updateProjectionMatrix();
    lastProgress = -1;
    if (ready) buildConnections();
  };
  const observer = new ResizeObserver(resize);
  observer.observe(host);
  resize();
  function separatedPosition(organ: Organ) {
    const x = mobile ? camera.right * 0.62 : Math.min(camera.right * 0.47, 1.0);
    return new T.Vector3(
      x * organ.side,
      mobile ? 1.45 - organ.row * 0.365 : 1.49 - organ.row * 0.375,
      0.15,
    );
  }
  function buildConnections() {
    for (const v of vessels) {
      scene.remove(v.mesh);
      v.mesh.geometry.dispose();
      v.mesh.material.dispose();
      ownedMaterials.delete(v.mesh.material);
    }
    vessels.length = 0;
    organs.forEach((organ, index) => {
      const tower = TOWERS.find((t) => t.id === organ.id)!;
      const count = tower.subprocessCount ?? tower.previewVesselCount;
      const origin = heart!.mesh.position
        .clone()
        .add(new T.Vector3(0, 0.02, 0.035));
      const geometry = buildVesselNetwork(
        origin,
        organ.center,
        separatedPosition(organ),
        count,
        organ.side,
        index * 31 + 7,
      );
      const uniforms = {
        uSeparation: { value: 0 },
        uTime: { value: 0 },
        uHoverOffset: { value: new T.Vector3() },
      };
      const material = new T.MeshPhysicalMaterial({
        color: '#ad566c',
        transparent: true,
        opacity: 0.46,
        roughness: 0.26,
        metalness: 0,
        depthWrite: false,
        side: T.DoubleSide,
        envMapIntensity: 0.25,
      });
      material.onBeforeCompile = (shader) => {
        Object.assign(shader.uniforms, uniforms);
        shader.vertexShader =
          'attribute vec3 expandedPosition; attribute vec3 expandedNormal; attribute float branchSeed; uniform float uSeparation; uniform vec3 uHoverOffset; varying float vPath; varying float vBranch;\n' +
          shader.vertexShader;
        shader.vertexShader = shader.vertexShader.replace(
          '#include <beginnormal_vertex>',
          '#include <beginnormal_vertex>\nobjectNormal = normalize(mix(normal,expandedNormal,uSeparation));',
        );
        shader.vertexShader = shader.vertexShader.replace(
          '#include <begin_vertex>',
          '#include <begin_vertex>\ntransformed = mix(position,expandedPosition,uSeparation) + uHoverOffset * smoothstep(0.45,1.0,uv.x); vPath=uv.x; vBranch=branchSeed;',
        );
        shader.fragmentShader =
          'uniform float uTime; varying float vPath; varying float vBranch;\n' +
          shader.fragmentShader;
        shader.fragmentShader = shader.fragmentShader.replace(
          '#include <color_fragment>',
          '#include <color_fragment>\nfloat flow = 0.5 + 0.5*sin(vPath*10.0-uTime*3.8+vBranch*2.3); diffuseColor.rgb *= 0.7+0.4*flow; diffuseColor.a *= 0.5+0.5*flow;',
        );
      };
      material.customProgramCacheKey = () => 'gfs-branching-vessels-v1';
      const mesh = new T.Mesh(geometry, material);
      mesh.frustumCulled = false;
      mesh.userData.subprocessCount = tower.subprocessCount;
      mesh.userData.previewCount = count;
      scene.add(mesh);
      ownedMaterials.add(material);
      vessels.push({ mesh, organ, uniforms });
    });
  }
  function arrange() {
    const eased = progress * progress * (3 - 2 * progress);
    const halfWidth = camera.right;
    const x = mobile ? halfWidth * 0.62 : Math.min(halfWidth * 0.47, 1.0);
    for (const o of organs) {
      const endY = mobile ? 1.45 - o.row * 0.365 : 1.49 - o.row * 0.375;
      const end = new T.Vector3(x * o.side, endY, 0.15);
      o.mesh.position.copy(o.center).lerp(end, eased);
      const endScale = o.expandedScale * (mobile ? 0.7 : 1);
      o.mesh.scale.setScalar(T.MathUtils.lerp(1, endScale, eased));
      o.mesh.rotation.y = T.MathUtils.lerp(0, o.side * -0.12, eased);
      o.basePosition.copy(o.mesh.position);
      o.baseScale = o.mesh.scale.x;
      o.mesh.updateMatrixWorld();
    }
    skullMaterial.opacity = T.MathUtils.lerp(0.42, 1, eased);
    // Transparent skull reveals the assembled brain. It becomes solid as the brain leaves.
    for (const v of vessels) {
      v.uniforms.uSeparation.value = eased;
    }
    projectLabels();
    lastProgress = progress;
  }
  function projectLabels() {
    const alpha = T.MathUtils.smoothstep(progress, 0.35, 0.85);
    for (const organ of organs) {
      const trigger = triggers.get(organ.id);
      if (!trigger) continue;
      temp.copy(organ.mesh.position).project(camera);
      const bw = Math.max(
        44,
        (organ.size.x * organ.mesh.scale.x * width) /
          (camera.right - camera.left),
      );
      const bh = Math.max(
        44,
        (organ.size.y * organ.mesh.scale.y * height) /
          (camera.top - camera.bottom),
      );
      trigger.style.width = `${bw}px`;
      trigger.style.height = `${bh}px`;
      trigger.style.transform = `translate3d(${((temp.x + 1) * width) / 2 - bw / 2}px,${((1 - temp.y) * height) / 2 - bh / 2}px,0)`;
      trigger.style.pointerEvents = progress > 0.7 ? 'auto' : 'none';
    }
    for (const o of organs) {
      if (!o.label) continue;
      temp.copy(o.mesh.position).project(camera);
      let px = ((temp.x + 1) * width) / 2,
        py = ((1 - temp.y) * height) / 2;
      const w = o.label.offsetWidth;
      if (mobile) {
        px -= w / 2;
        py +=
          Math.max(
            25,
            (((o.size.y * o.mesh.scale.y) / (camera.top - camera.bottom)) *
              height) /
              2,
          ) + 7;
      } else {
        const gap =
          Math.max(
            32,
            (((o.size.x * o.mesh.scale.x) / (camera.right - camera.left)) *
              width) /
              2,
          ) + 16;
        px += o.side < 0 ? -gap - w : gap;
        py -= 15;
      }
      px = T.MathUtils.clamp(px, 10, width - w - 10);
      o.label.style.transform = `translate3d(${px}px,${py}px,0)`;
      o.label.style.opacity = String(alpha);
      o.label.setAttribute('aria-hidden', alpha < 0.1 ? 'true' : 'false');
    }
  }
  async function load() {
    try {
      callbacks.onProgress('Loading the anatomy');
      const [manifestResponse, response] = await Promise.all([
        fetch('/data/anatomy/atlas.json', { signal: abort.signal }),
        fetch('/data/anatomy/body.bin.gz', { signal: abort.signal }),
      ]);
      if (!manifestResponse.ok || !response.ok)
        throw new Error(
          'The anatomy could not be loaded. Please check your connection and try again.',
        );
      const atlas = (await manifestResponse.json()) as {
        parts: Part[];
        bytes: number;
      };
      const zipped = await response.arrayBuffer();
      if (disposed) return;
      const signature = new Uint8Array(zipped, 0, 2);
      const buffer =
        signature[0] === 31 && signature[1] === 139
          ? await new Response(
              new Blob([zipped])
                .stream()
                .pipeThrough(new DecompressionStream('gzip')),
            ).arrayBuffer()
          : zipped;
      if (buffer.byteLength !== atlas.bytes)
        throw new Error(
          'The anatomy download was incomplete. Please try again.',
        );
      if (disposed) return;
      callbacks.onProgress('Assembling bones and organs');
      const grouped = new Map<string, T.BufferGeometry[]>();
      for (const p of atlas.parts) {
        const g = new T.BufferGeometry();
        g.setAttribute(
          'position',
          new T.BufferAttribute(
            new Float32Array(buffer, p.positions, p.vertexCount * 3),
            3,
          ),
        );
        g.setAttribute(
          'normal',
          new T.BufferAttribute(
            new Int16Array(buffer, p.normals, p.vertexCount * 3),
            3,
            true,
          ),
        );
        g.setIndex(
          new T.BufferAttribute(
            new Uint32Array(buffer, p.indices, p.indexCount),
            1,
          ),
        );
        const group =
          p.group === 'skeleton' && p.bounds[0][1] > 1.45 ? 'skull' : p.group;
        const list = grouped.get(group) ?? [];
        list.push(g);
        grouped.set(group, list);
      }
      for (const [id, pieces] of grouped) {
        const geometry = mergeGeometries(pieces, false);
        pieces.forEach((p) => p.dispose());
        if (!geometry) throw new Error('Could not assemble anatomy.');
        ownedGeometries.add(geometry);
        geometry.computeBoundingBox();
        geometry.computeBoundingSphere();
        if (id === 'skeleton' || id === 'skull') {
          continue;
        }
        const center = geometry.boundingBox!.getCenter(new T.Vector3());
        const size = geometry.boundingBox!.getSize(new T.Vector3());
        geometry.translate(-center.x, -center.y, -center.z);
        const tower = TOWERS.find((t) => t.id === id);
        let material;
        if (id === 'brain') {
          // Holographic blue medical scan style
          material = new T.MeshPhysicalMaterial({
            color: '#001133',
            emissive: '#00aaff',
            emissiveIntensity: 0.65,
            roughness: 0.15,
            metalness: 0.9,
            transparent: true,
            opacity: 0.45,
            depthWrite: false, 
            side: T.DoubleSide,
            clearcoat: 1.0,
          });
        } else {
          // Default style for heart and other organs
          material = new T.MeshPhysicalMaterial({
            color: id === 'heart' ? '#9e244c' : (tower?.color ?? '#be9e94'),
            roughness: 0.43,
            metalness: 0.02,
            clearcoat: 0.17,
            envMapIntensity: 0.28,
            side: T.DoubleSide,
          });
        }
        ownedMaterials.add(material);
        const mesh = new T.Mesh(geometry, material);
        mesh.position.copy(center);
        mesh.userData.organ = id;
        mesh.userData.towerId = id === 'heart' ? 'gfs-kl' : tower?.name;
        const organ: Organ = {
          id,
          mesh,
          center,
          size,
          expandedScale: Math.min(0.235 / size.y, 0.29 / size.x, 2.2),
          label: labelHost.querySelector(`[data-organ="${id}"]`),
          side: tower?.side === 'left' ? -1 : 1,
          row: tower?.row ?? 0,
          hover: 0,
          basePosition: center.clone(),
          baseScale: 1,
        };
        scene.add(mesh);
        if (id === 'heart') {
          heart = organ;
          mesh.position.z += 0.085;
          mesh.renderOrder = 2;
          material.depthTest = false;
          mesh.scale.setScalar(1.13);
        } else {
          organs.push(organ);
          organGeometries.set(id, geometry);
        }
      }
      if (!heart || organs.length !== 8)
        throw new Error(
          'Some organs are missing from the anatomy. Please try again.',
        );
      const gltfLoader = new GLTFLoader();
      callbacks.onProgress('Loading 3D Body Model...'); 

      gltfLoader.load('/anatomy/human-body.glb', (gltf) => {
          const body = gltf.scene;

          const finalBodyMaterial = new T.MeshStandardMaterial({
              color: '#050011',
              emissive: '#7a0026',
              emissiveIntensity: 0.6,
              roughness: 0.3,
              metalness: 0.8,
              transparent: true,
              opacity: 0.25,
              side: T.DoubleSide,       
              depthWrite: false
          });

          body.traverse((child) => {
              if ((child as T.Mesh).isMesh) {
                  const mesh = child as T.Mesh;
                  mesh.geometry.computeVertexNormals(); 
                  mesh.material = finalBodyMaterial;
                  mesh.frustumCulled = false; 
              }
          });

          const box = new T.Box3().setFromObject(body);
          const size = box.getSize(new T.Vector3());
          const scaleFactor = 2.15 / (size.y || 1); 
          body.scale.setScalar(scaleFactor);
          body.updateMatrixWorld(true); 

          const scaledBox = new T.Box3().setFromObject(body);
          const scaledCenter = scaledBox.getCenter(new T.Vector3());

          body.position.x += (0 - scaledCenter.x) - 0.25; 
          body.position.y += (0.88 - scaledCenter.y) - 0.45; 
          body.position.z += (-0.08 - scaledCenter.z) - 0.05;

          scene.add(body);

          // Colleague's original setup executes after body is fully loaded
          buildConnections();
          arrange();
          ready = true;
          for (const [id, count] of pendingCounts) {
            const organ = organs.find((o) => o.id === id);
            if (organ) prepareNeurons(organ, count);
          }
          pendingCounts.clear();
          callbacks.onReady();
      });
    } catch (e) {
      if (!disposed)
        callbacks.onError(
          e instanceof Error
            ? e.message
            : 'The anatomy could not be loaded. Please try again.',
        );
    }
  }
  // Process counts requested before the anatomy finished loading.
  const pendingCounts = new Map<string, number>();
  void load();
  const tick = (now: number) => {
    if (disposed) return;
    const dt = Math.min((now - lastTime) / 1000, 0.05);
    lastTime = now;
    if (!document.hidden) {
      // Assembly follows the wheel on a soft damp so the organs drift apart and back.
      const next =
        document.documentElement.dataset.input === 'keyboard'
          ? target
          : T.MathUtils.damp(progress, target, reduced ? 20 : 6.5, dt);
      progress = Math.abs(next - target) < 0.0001 ? target : next;
      if (ready && Math.abs(lastProgress - progress) > 0.00001) arrange();
      // Time keeps running under reduced motion so the heart still beats (more softly).
      if (!paused) time += dt;
      if (ready && heart) {
        // A double contraction, then relaxation. Flow travels even when the slider is still.
        const cycle = (time % 1.02) / 1.02;
        // Visible lub-dub: a strong first contraction, a softer second, then rest.
        const beat =
          Math.exp(-Math.pow((cycle - 0.12) / 0.07, 2)) * 0.12 +
          Math.exp(-Math.pow((cycle - 0.32) / 0.085, 2)) * 0.06;
        const scale = 1.13 + (reduced ? beat * 0.5 : beat);
        heart.mesh.scale.set(scale, scale * 0.995, scale);
        if (heartWord) {
          heartWord.style.transform = `scale(${1 + (reduced ? 0 : beat * 0.65)})`;
          heartWord.style.opacity = String(reduced ? 1 : 0.86 + beat * 2);
        }
        for (const organ of organs) {
          organ.hover = T.MathUtils.damp(
            organ.hover,
            organ.id === hovered ? 1 : 0,
            reduced ? 14 : 6,
            dt,
          );
          const hover = organ.hover;
          // Hovered organ lifts, floats on a slow bob and gently sways as if suspended.
          organ.mesh.position.copy(organ.basePosition);
          if (!reduced) {
            organ.mesh.position.y +=
              hover * (0.03 + Math.sin(time * 2.2) * 0.014);
            organ.mesh.position.x += hover * Math.sin(time * 1.3) * 0.008;
          }
          organ.mesh.scale.setScalar(organ.baseScale * (1 + hover * 0.11));
          organ.mesh.rotation.y =
            organ.side * -0.12 * progress +
            (reduced ? 0 : hover * Math.sin(time * 1.6) * 0.2);
          organ.mesh.rotation.z = reduced
            ? 0
            : hover * Math.sin(time * 1.4) * 0.05;
          organ.mesh.updateMatrixWorld();
        }
        for (const v of vessels) {
          v.uniforms.uTime.value = time;
          v.uniforms.uHoverOffset.value
            .copy(v.organ.mesh.position)
            .sub(v.organ.basePosition);
          v.mesh.material.opacity = 0.42 + v.organ.hover * 0.22;
        }
      }
      if (!anchor) renderer.render(scene, camera);
      renderFlight(now);
    }
    frame = requestAnimationFrame(tick);
  };
  frame = requestAnimationFrame(tick);
  const onContextLost = (event: Event) => {
    event.preventDefault();
    callbacks.onError(
      'The 3D view was interrupted. Try again to reload the anatomy.',
    );
  };
  renderer.domElement.addEventListener('webglcontextlost', onContextLost);
  return {
    focusOrgan,
    setProcessCount(organId, count) {
      const organ = organs.find((o) => o.id === organId);
      if (!ready || !organ) {
        pendingCounts.set(organId, count);
        return;
      }
      prepareNeurons(organ, Math.max(0, count));
      if (flyMesh && focusedOrgan === organId)
        activeNeurons().forEach((node) => flyMesh!.add(node));
    },
    setView(y, p, z, j) {
      yaw = y;
      pitch = p;
      zoom = T.MathUtils.clamp(z, 0.65, 2.2);
      journey = T.MathUtils.clamp(j, 0, 1);
    },
    setProgress(value) {
      target = T.MathUtils.clamp(value, 0, 1);
    },
    setPaused(value) {
      paused = value;
    },
    destroy() {
      flyEnvironment?.dispose();
      flyRenderer?.dispose();
      flyRenderer?.forceContextLoss();
      flyRenderer?.domElement.remove();
      host.removeEventListener('pointerup', onClick);
      disposed = true;
      abort.abort();
      cancelAnimationFrame(frame);
      observer.disconnect();
      preference.removeEventListener('change', onPreference);
      host.removeEventListener('pointermove', onPointer);
      host.removeEventListener('pointerleave', clearHover);
      renderer.domElement.removeEventListener(
        'webglcontextlost',
        onContextLost,
      );
      for (const v of vessels) {
        v.mesh.geometry.dispose();
      }
      ownedGeometries.forEach((g) => g.dispose());
      ownedMaterials.forEach((m) => m.dispose());
      environment.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
      labels.forEach((l) => {
        l.style.opacity = '0';
      });
    },
  };
}
