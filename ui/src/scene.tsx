import { Canvas, type ThreeEvent, useFrame, useThree } from '@react-three/fiber'
import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import type { GridState, Line, PlantPreset } from './api'
import type { Allocation, Campus } from './missionApi'
import { RegionalTerrain, regionalGeography } from './RegionalTerrain'

type Point = [number, number, number]
type SceneState = {buses: GridState['buses']; lines: GridState['lines']; compute: GridState['compute']; network?: string; site?: GridState['site']; preset?: GridState['preset']}
type Pick = {type:'bus' | 'line'; id: number}
const up = new THREE.Vector3(0, 1, 0)
const ground = new THREE.Plane(up, -0.1)
const copper = '#c04934'
const boardWidth = 25
const boardDepth = 18

export function layout(state: SceneState) {
  const positions = new Map<number, Point>()
  const map = regionalGeography(state.network)
  if (map?.anchors.length === state.buses.length) {
    state.buses.forEach((bus, i) => {
      const [x,z] = map.anchors[i]
      positions.set(bus.id, [x,.1,z])
    })
    return positions
  }
  if (state.buses.length && state.buses.every(bus => bus.position)) {
    for (const bus of state.buses) {
      if (bus.position) positions.set(bus.id, [bus.position[0], .1, bus.position[1]])
    }
    return positions
  }
  const points = state.buses.map((bus, i) => ({
    id: bus.id, x: 6 * Math.cos(i * 2.399963), z: 6 * Math.sin(i * 2.399963),
  }))
  const lookup = new Map(points.map((p, i) => [p.id, i]))
  for (let iteration = 0; iteration < 105; iteration++) {
    const force = points.map(p => ({x: -p.x * 0.018, z: -p.z * 0.018}))
    for (let i = 0; i < points.length; i++) for (let j = i + 1; j < points.length; j++) {
      const dx = points[i].x - points[j].x, dz = points[i].z - points[j].z
      const scale = 0.42 / Math.max(0.2, Math.pow(dx * dx + dz * dz, 1.3))
      force[i].x += dx * scale
      force[i].z += dz * scale
      force[j].x -= dx * scale
      force[j].z -= dz * scale
    }
    for (const line of state.lines) {
      const i = lookup.get(line.from), j = lookup.get(line.to)
      if (i === undefined || j === undefined) continue
      const dx = points[i].x - points[j].x, dz = points[i].z - points[j].z
      const length = Math.hypot(dx, dz) || 1
      const scale = (length - 2.55) * 0.033 / length
      force[i].x -= dx * scale
      force[i].z -= dz * scale
      force[j].x += dx * scale
      force[j].z += dz * scale
    }
    points.forEach((p, i) => { p.x += force[i].x; p.z += force[i].z })
  }
  const scale = 7.6 / Math.max(1, ...points.map(p => Math.max(Math.abs(p.x), Math.abs(p.z))))
  points.forEach(p => positions.set(p.id, [p.x * scale * boardWidth / boardDepth, 0.1, p.z * scale]))
  return positions
}

function CameraRig({ dragging }: {dragging: boolean}) {
  const {camera, gl, size} = useThree()
  const controls = useRef<OrbitControls>(null)
  useEffect(() => {
    const control = new OrbitControls(camera, gl.domElement)
    control.target.set(0, 0, 0)
    control.enableDamping = true
    control.enablePan = true
    control.enableRotate = false
    control.mouseButtons.LEFT = THREE.MOUSE.PAN
    control.touches.ONE = THREE.TOUCH.PAN
    control.touches.TWO = THREE.TOUCH.DOLLY_PAN
    control.minZoom = 9
    control.maxZoom = 72
    controls.current = control
    return () => control.dispose()
  }, [camera, gl])
  useEffect(() => {
    camera.zoom = Math.min(size.width / (boardWidth + 2.5), size.height / (boardDepth + 2.5))
    camera.updateProjectionMatrix()
  }, [camera, size.height, size.width])
  useEffect(() => { if (controls.current) controls.current.enabled = !dragging }, [dragging])
  useFrame(() => controls.current?.update())
  return null
}

function Box({ position, size, color }: {position: Point; size: Point; color: string}) {
  return <mesh position={position} castShadow receiveShadow>
    <boxGeometry args={size}/><meshStandardMaterial color={color} roughness={0.92}/>
  </mesh>
}

function Stack({ position, top, bottom, height, color }: {
  position: Point; top: number; bottom: number; height: number; color: string
}) {
  return <mesh position={position} castShadow>
    <cylinderGeometry args={[top,bottom,height,16]}/>
    <meshStandardMaterial color={color} roughness={.88}/>
  </mesh>
}

const coolingProfile = [
  [.15,0], [.125,.11], [.105,.25], [.095,.40], [.105,.56], [.13,.67],
].map(([radius,height]) => new THREE.Vector2(radius,height))

function Plant({ preset, position, onPick }: {preset: PlantPreset; position: Point; onPick: () => void}) {
  return <group position={position} onClick={event => {event.stopPropagation(); onPick()}}>
    <Box position={[0,.015,0]} size={[3.85,.075,2.65]} color={preset === 'surry' ? '#98aba3' : '#a8aa92'}/>
    {preset === 'senoko' && <>
      {[-1,0,1].map(x => <group key={x} position={[x,0,0]}>
        <Box position={[0,.30,.22]} size={[.77,.52,1.2]} color="#628087"/>
        <Box position={[0,.58,.24]} size={[.83,.08,1.25]} color="#8fafad"/>
        <Box position={[0,.22,-.6]} size={[.50,.40,.39]} color="#c0af89"/>
        <Stack position={[0,.56,-.79]} top={.085} bottom={.13} height={1.02} color="#f0d8ad"/>
        <Stack position={[0,1.12,-.79]} top={.095} bottom={.095} height={.12} color="#b45738"/>
      </group>)}
    </>}
    {preset === 'surry' && <>
      <Box position={[0,.063,-1.01]} size={[3.8,.022,.57]} color="#72a4ad"/>
      <Box position={[0,.08,-.8]} size={[3.76,.02,.025]} color="#9fc3c0"/>
      {[-.65,.65].map(x => <group key={x} position={[x,0,-.13]}>
        <Stack position={[0,.28,0]} top={.43} bottom={.43} height={.46} color="#dedfd0"/>
        <mesh position={[0,.49,0]} castShadow>
          <sphereGeometry args={[.44,20,12,0,Math.PI * 2,0,Math.PI / 2]}/>
          <meshStandardMaterial color="#d5e1d9" roughness={.9}/>
        </mesh>
      </group>)}
      <Box position={[0,.24,.66]} size={[2.46,.36,.64]} color="#8a9d8c"/>
      <Box position={[0,.45,.67]} size={[2.56,.08,.69]} color="#b3c3ac"/>
    </>}
    {preset === 'drax' && <>
      {Array.from({length:12}, (_, i) => <group key={i} position={[-1.3 + (i % 6) * .38,.06,-.92 + Math.floor(i / 6) * .49]}>
        <mesh castShadow>
          <latheGeometry args={[coolingProfile,14]}/>
          <meshStandardMaterial color="#ad9885" side={THREE.DoubleSide} roughness={.96}/>
        </mesh>
        <Stack position={[0,.1,0]} top={.126} bottom={.13} height={.04} color="#776f66"/>
      </group>)}
      <Box position={[.04,.29,.55]} size={[2.9,.46,.78]} color="#8b7d72"/>
      <Box position={[.04,.54,.53]} size={[2.98,.08,.86]} color="#b38a74"/>
      <Stack position={[1.47,.71,-.32]} top={.11} bottom={.16} height={1.3} color="#a9a393"/>
    </>}
  </group>
}

function sitePosition(points: Map<number, Point>, anchorId: number) {
  const anchor = points.get(anchorId)
  if (!anchor) return undefined
  const candidates = Array.from({length:16}, (_, index) => {
    const angle = index * Math.PI / 8
    const distance = index % 2 ? 2.9 : 2.4
    const x = THREE.MathUtils.clamp(anchor[0] + Math.cos(angle) * distance,-10.5,10.5)
    const z = THREE.MathUtils.clamp(anchor[2] + Math.sin(angle) * distance,-7.5,7.5)
    const clearance = Math.min(...[...points.values()].map(([px,,pz]) => Math.hypot((x - px) * .75,z - pz)))
    return {position:[x,.025,z] as Point, clearance}
  })
  return candidates.sort((a,b) => b.clearance - a.clearance)[0].position
}

export function lineStatus(line: Line) {
  if (!line.active) return {name:'Open', color:'#a3a59a'}
  if (line.loading === null) return {name:'Unobserved', color:'#a3a59a'}
  if (line.loading > 1) return {name:'Over limit', color:'#d54a30'}
  if (line.loading >= .8) return {name:'Near limit', color:'#d99130'}
  return {name:'In service', color:'#327c72'}
}

function Branch({ line, start, end, offset, selected, limiting, ghost, onPick }: {
  line: Line; start: Point; end: Point; offset: number; selected: boolean; limiting?: boolean; ghost?: number | null; onPick: () => void
}) {
  const {color} = lineStatus(line)
  const marker = useRef<THREE.Mesh>(null)
  const curve = useMemo(() => {
    const dx = end[0] - start[0], dz = end[2] - start[2]
    const distance = Math.hypot(dx, dz) || 1
    return new THREE.QuadraticBezierCurve3(
      new THREE.Vector3(start[0], .37, start[2]),
      new THREE.Vector3((start[0]+end[0])/2 - offset * dz / distance, .17, (start[2]+end[2])/2 + offset * dx / distance),
      new THREE.Vector3(end[0], .37, end[2]),
    )
  }, [start, end, offset])
  const length = curve.getLength()
  useFrame(({clock}) => {
    if (!marker.current) return
    const backwards = (line.flow ?? 0) < 0
    const time = (clock.elapsedTime * (0.2 + Math.min(0.48, Math.abs(line.flow ?? 0) / 500)) + line.id * .11) % 1
    const t = backwards ? 1 - time : time
    marker.current.position.copy(curve.getPoint(t))
    const dir = curve.getTangent(t).multiplyScalar(backwards ? -1 : 1)
    marker.current.quaternion.setFromUnitVectors(up, dir)
  })
  const points = useMemo(() => curve.getPoints(12), [curve])
  const pick = (event: ThreeEvent<MouseEvent>) => {event.stopPropagation(); onPick()}
  return <group>
    <mesh onClick={pick}>
      <tubeGeometry args={[curve, 12, .15, 5, false]}/>
      <meshBasicMaterial transparent opacity={0} depthWrite={false}/>
    </mesh>
    {limiting && <mesh>
      <tubeGeometry args={[curve, 12, .135, 7, false]}/>
      <meshBasicMaterial color="#d6452d" transparent opacity={.36} depthWrite={false}/>
    </mesh>}
    {ghost != null && ghost >= .8 && points.filter((_, i) => i % 2 === 1).map((point, i) => <mesh key={i} position={[point.x, .67, point.z]}>
      <sphereGeometry args={[ghost > 1 ? .066 : .046, 6, 4]}/><meshBasicMaterial color={ghost > 1 ? '#a84c92' : '#6961b6'}/>
    </mesh>)}
    {line.active ? <mesh>
      <tubeGeometry args={[curve, 12, selected || limiting ? .061 : .032, 5, false]}/>
      <meshStandardMaterial color={limiting ? '#e44227' : color} emissive={limiting ? '#e44227' : color} emissiveIntensity={limiting ? .56 : selected ? .21 : .03}/>
    </mesh> : points.filter((_,i) => i % 2 === 0).map((p,i) => <mesh key={i} position={p}>
      <sphereGeometry args={[selected ? .053 : .032, 5, 4]}/><meshStandardMaterial color={color}/>
    </mesh>)}
    {line.active && length > .2 && Math.abs(line.flow ?? 0) > .01 && <mesh ref={marker}>
      <coneGeometry args={[.068, .16, 5]}/><meshBasicMaterial color={color}/>
    </mesh>}
  </group>
}

function Landscape({ positions, site, network }: {positions: Map<number, Point>; site?: Point; network?: string}) {
  const buildings = useRef<THREE.InstancedMesh>(null)
  const groves = useRef<THREE.InstancedMesh>(null)
  const island = network === "singapore"
  const mapped = Boolean(regionalGeography(network))
  const parcels = useMemo(() => {
    const nodes = [...positions.values()]
    return Array.from({length: mapped ? 0 : 200}, (_, i) => {
      const x = Math.sin(i * 91.31 + 2) * (boardWidth / 2 - .7)
      const z = Math.sin(i * 62.23 + 4) * (boardDepth / 2 - .7)
      const height = .12 + (i % 9) * .045
      return {x, z, height, tree: i % 3 === 0, i}
    }).filter(p => nodes.every(([x,,z]) => Math.hypot(x - p.x, z - p.z) > .82) &&
      (!site || Math.abs(site[0] - p.x) > 2.04 || Math.abs(site[2] - p.z) > 1.47))
  }, [positions, site, mapped])
  useEffect(() => {
    const dummy = new THREE.Object3D(), color = new THREE.Color()
    let house = 0, tree = 0
    parcels.forEach(p => {
      const ref = p.tree ? groves.current : buildings.current
      if (!ref) return
      dummy.position.set(p.x, p.height / 2 + .06, p.z)
      dummy.rotation.set(0, p.i % 2 ? 0 : Math.PI / 2, 0)
      dummy.scale.set(.22 + (p.i % 4) * .06, p.height, .25 + (p.i % 3) * .1)
      dummy.updateMatrix()
      const idx = p.tree ? tree++ : house++
      ref.setMatrixAt(idx, dummy.matrix)
      color.set(p.tree ? ['#9da68a','#a7b293','#929f82'][p.i % 3] : ['#d4cbbb','#c0c7b5','#c9baa4','#bdc7c0'][p.i % 4])
      ref.setColorAt(idx, color)
    })
    for (const [ref, count] of [[buildings.current, house], [groves.current, tree]] as const) {
      if (ref) { ref.count = count; ref.instanceMatrix.needsUpdate = true; if (ref.instanceColor) ref.instanceColor.needsUpdate = true; ref.computeBoundingSphere() }
    }
  }, [parcels])
  return <>
    <Box position={[0,-.34,0]} size={[boardWidth,.6,boardDepth]} color="#adada0"/>
    <Box position={[0,-.04,0]} size={[boardWidth,.1,boardDepth]} color={island ? "#a9ced2" : "#dbded1"}/>
    <RegionalTerrain network={network}/>
    <gridHelper args={[boardDepth, boardDepth, '#c9cdbd', '#c9cdbd']} position={[0,.021,0]} scale={[boardWidth / boardDepth,1,1]}/>
    <instancedMesh ref={buildings} args={[undefined, undefined, 200]} castShadow receiveShadow>
      <boxGeometry args={[1,1,1]}/><meshStandardMaterial roughness={.9}/>
    </instancedMesh>
    <instancedMesh ref={groves} args={[undefined, undefined, 200]} castShadow>
      <icosahedronGeometry args={[.66,0]}/><meshStandardMaterial roughness={1}/>
    </instancedMesh>
  </>
}

function MapLabel({text, color, position, wide = false}: {text: string; color: string; position: Point; wide?: boolean}) {
  const texture = useMemo(() => {
    const canvas = document.createElement('canvas')
    canvas.width = wide ? 360 : 160
    canvas.height = 64
    const context = canvas.getContext('2d')
    if (!context) return null
    context.fillStyle = '#f7f8f0'
    context.beginPath()
    context.roundRect(3, 4, canvas.width - 6, 56, 11)
    context.fill()
    context.fillStyle = color
    context.font = wide ? '600 29px system-ui' : '600 35px monospace'
    context.textAlign = 'center'
    context.textBaseline = 'middle'
    context.fillText(text, canvas.width / 2, 33, canvas.width - 16)
    const label = new THREE.CanvasTexture(canvas)
    label.colorSpace = THREE.SRGBColorSpace
    return label
  }, [text,color,wide])
  useEffect(() => () => texture?.dispose(), [texture])
  return texture && <sprite position={position} scale={[wide ? 2.7 : 1.22,.49,1]}>
    <spriteMaterial map={texture} transparent depthWrite={false}/>
  </sprite>
}

function CampusModel({campus, point, onPick}: {campus: Campus; point: Point; onPick: () => void}) {
  const length = Math.min(2, 1.25 + Math.max(0, campus.capacity_mw) / 140)
  return <group position={point} onClick={event => {event.stopPropagation(); onPick()}}>
    <Box position={[0,.06,0]} size={[length+.24,.11,1.04]} color="#36665c"/>
    <Box position={[0,.34,0]} size={[length,.49,.84]} color="#30665b"/>
    <Box position={[0,.63,0]} size={[length+.07,.10,.90]} color="#e7f2dd"/>
    {[-.30,0,.30].map(x => <Box key={x} position={[x,.7,0]} size={[.14,.07,.55]} color="#68a389"/>)}
    {campus.battery_enabled && <group position={[0,0,.64]}>
      <Box position={[0,.18,0]} size={[length,.32,.26]} color="#4f586b"/>
      <Box position={[-length/2 + length * Math.max(.03, Math.min(1, campus.soc))/2,.36,0]} size={[length * Math.max(.03, Math.min(1, campus.soc)),.06,.18]} color="#b8d960"/>
    </group>}
    <MapLabel text={`B${campus.bus}`} color="#235544" position={[0,.82,-.77]}/>
    {campus.backlog_mwh > .001 && <mesh position={[length/2,.77,-.40]}>
      <sphereGeometry args={[.12, 12, 8]}/><meshBasicMaterial color="#de963d"/>
    </mesh>}
  </group>
}

function CampusGhost({allocation, point, onPick}: {allocation: Allocation; point: Point; onPick: () => void}) {
  const size: Point = [Math.min(2, 1.2 + allocation.mw / 125), .65, 1.05]
  const edges = useMemo(() => {
    const box = new THREE.BoxGeometry(...size)
    const lines = new THREE.EdgesGeometry(box)
    box.dispose()
    return lines
  }, [size[0]])
  useEffect(() => () => edges.dispose(), [edges])
  return <group position={[point[0], point[1]+.73, point[2]]} onClick={event => {event.stopPropagation(); onPick()}}>
    <mesh>
      <boxGeometry args={size}/>
      <meshBasicMaterial color="#8a72d6" transparent opacity={.3} depthWrite={false}/>
    </mesh>
    <lineSegments geometry={edges}><lineBasicMaterial color="#6752ad"/></lineSegments>
    <MapLabel text={`B${allocation.bus}`} color="#6752ad" position={[0,.4,-.77]}/>
  </group>
}

function SceneContents({ state, selected, onPick, onMove, canMove, campuses, ghost, limiting }: {
  state: SceneState; selected?: Pick; onPick: (pick: Pick) => void; onMove?: (bus: number) => void; canMove?: boolean
  campuses?: Campus[]; ghost?: {allocation: Allocation[]; loading: Array<number | null>}; limiting?: number
}) {
  const topology = (state.network ?? '') + ':' + state.buses.map(b => b.id).join(',') + ':' + state.lines.map(l => `${l.from}-${l.to}`).join(',')
  const points = useMemo(() => layout(state), [topology])
  const anchor = state.site?.anchor_bus
  const site = useMemo(() => anchor === undefined ? undefined : sitePosition(points, anchor), [anchor, points])
  const offsets = useMemo(() => {
    const parallel = new Map<string, Line[]>()
    for (const line of state.lines) {
      const key = `${Math.min(line.from,line.to)}-${Math.max(line.from,line.to)}`
      parallel.set(key, [...(parallel.get(key) ?? []), line])
    }
    return new Map([...parallel.values()].flatMap(group => group.map((line, index) => [
      line.id,
      (index - (group.length - 1) / 2) * .72 * (line.from < line.to ? 1 : -1),
    ] as const)))
  }, [topology])
  const [drag, setDrag] = useState<{point: Point; bus: number} | null>(null)
  const dragging = useRef(false)
  const nearest = (event: ThreeEvent<PointerEvent>) => {
    const position = event.ray.intersectPlane(ground, new THREE.Vector3())
    if (!position) return null
    let best = Infinity, bus = state.compute.bus
    state.buses.forEach(item => {
      const point = points.get(item.id)
      if (item.eligible === false || !point) return
      const distance = Math.hypot(point[0] - position.x, point[2] - position.z)
      if (distance < best) { best = distance; bus = item.id }
    })
    return {point: [position.x, .1, position.z] as Point, bus}
  }
  const campus = drag?.point ?? points.get(state.compute.bus)
  return <>
    <color attach="background" args={['#eeede5']}/>
    <ambientLight intensity={1.6}/>
    <hemisphereLight args={['#f8f2dd', '#a8a596', 1.45]}/>
    <directionalLight position={[-7,16,8]} intensity={2.1} castShadow shadow-mapSize={[2048,2048]} shadow-camera-left={-12} shadow-camera-right={12} shadow-camera-bottom={-12} shadow-camera-top={12} shadow-normalBias={.035}/>
    <CameraRig dragging={drag !== null}/>
    <Landscape positions={points} site={site} network={state.network}/>
    {state.preset && site && anchor !== undefined && <Plant preset={state.preset} position={site} onPick={() => onPick({type:'bus', id:anchor})}/>}
    {state.lines.map(line => {
      const start = points.get(line.from), end = points.get(line.to)
      return start && end ? <Branch key={line.id} line={line} start={start} end={end} offset={offsets.get(line.id) ?? 0} selected={selected?.type === 'line' && selected.id === line.id} limiting={limiting === line.id} ghost={ghost?.loading[state.lines.indexOf(line)]} onPick={() => onPick({type:'line', id:line.id})}/> : null
    })}
    {state.buses.map(bus => {
      const p = points.get(bus.id)
      if (!p) return null
      const highlighted = (selected?.type === 'bus' && selected.id === bus.id) || drag?.bus === bus.id
      const highGen = bus.generation > .1
      const outside = bus.voltage !== null && (bus.voltage < .9 || bus.voltage > 1.1)
      return <group key={bus.id} position={p} onClick={e => {e.stopPropagation(); onPick({type:'bus', id:bus.id})}}>
        <Box position={[0,.04,0]} size={[.66,.13,.66]} color={highlighted ? copper : campuses !== undefined && bus.eligible ? '#75a173' : '#999c8d'}/>
        <Box position={[0,.20,0]} size={[.42,.24,.43]} color={bus.voltage === 0 ? '#75776e' : outside ? '#d99130' : '#e9e4d4'}/>
        <Box position={[0,.34,0]} size={[.24,.10,.24]} color={highGen ? '#4a6964' : '#65736b'}/>
        {bus.name && <MapLabel text={bus.name} color="#315644" position={[0,.55,-.72]} wide/>}
        {highGen && <Box position={[.27,.41,0]} size={[.11,.65,.13]} color="#8b9c8d"/>}
      </group>
    })}
    {campuses?.filter(item => item.capacity_mw > 0).map(item => {
      const point = points.get(item.bus)
      return point && <CampusModel key={item.bus} campus={item} point={point} onPick={() => onPick({type:'bus', id:item.bus})}/>
    })}
    {ghost?.allocation.filter(item => item.mw > 0).map(item => {
      const point = points.get(item.bus)
      return point && <CampusGhost key={item.bus} allocation={item} point={point} onPick={() => onPick({type:'bus', id:item.bus})}/>
    })}
    {campuses === undefined && campus && <group position={campus} onPointerDown={event => {
      event.stopPropagation()
      if (!canMove) return
      const target = event.target as Element
      target.setPointerCapture(event.pointerId)
      dragging.current = true
      setDrag({point: campus, bus: state.compute.bus})
    }} onPointerMove={event => {
      if (!dragging.current) return
      event.stopPropagation()
      const target = nearest(event)
      if (target) setDrag(target)
    }} onPointerUp={event => {
      event.stopPropagation()
      if (!dragging.current) { onPick({type:'bus', id:state.compute.bus}); return }
      const captured = event.target as Element
      captured.releasePointerCapture(event.pointerId)
      dragging.current = false
      const target = nearest(event)
      setDrag(null)
      if (target && target.bus !== state.compute.bus) onMove?.(target.bus)
      else onPick({type:'bus', id:state.compute.bus})
    }} onLostPointerCapture={() => {dragging.current = false; setDrag(null)}}>
      <Box position={[0,.15,0]} size={[.98,.25,.87]} color="#a03728"/>
      <Box position={[0,.46,0]} size={[.76,.46,.67]} color="#e56646"/>
      <Box position={[0,.72,0]} size={[.81,.12,.71]} color="#f4c1a7"/>
      {[-.22,0,.22].map(x => <Box key={x} position={[x,.8,0]} size={[.12,.055,.43]} color="#b65842"/>)}
    </group>}
  </>
}

export function GridScene({fallback, ...props}: Parameters<typeof SceneContents>[0] & {fallback?: React.ReactNode}) {
  return <Canvas fallback={fallback} shadows dpr={[1,1.5]} orthographic camera={{position:[0,28,.001], zoom:33, near:.1, far:120}}>
    <SceneContents {...props}/>
  </Canvas>
}
