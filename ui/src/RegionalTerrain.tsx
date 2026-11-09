import { useEffect, useMemo } from 'react'
import * as THREE from 'three'
import geography from './regionalGeography.json'

const colors: Record<string, string> = {
  island:'#dce6c7', virginia:'#dce6cb', neighbor:'#e9e5ce', water:'#91bcc9',
}
const boundary = '#adbca0'

export function regionalGeography(network?: string) {
  return network === 'singapore' || network === 'virginia' ? geography[network] : undefined
}

export function TerrainDiagram({network}: {network?: string}) {
  const map = regionalGeography(network)
  if (!map) return null
  return <>
    {map.polygons.map(({rings,kind},i) => <path key={i} fill={colors[kind]} fillRule="evenodd"
      stroke={kind === 'water' ? 'none' : boundary} strokeWidth=".035"
      d={rings.map(ring => 'M' + ring.map(point => point.join(' ')).join('L') + 'Z').join(' ')}/>)}
  </>
}

export function RegionalTerrain({network}: {network?: string}) {
  const meshes = useMemo(() => regionalGeography(network)?.polygons.map(({rings,kind}) => {
    const points = (ring: number[][]) => ring.map(([x,z]) => new THREE.Vector2(x,-z))
    const [outer,...holes] = rings
    const shape = new THREE.Shape(points(outer))
    shape.holes = holes.map(ring => new THREE.Path(points(ring)))
    const geometry = new THREE.ShapeGeometry(shape)
    const outline = new THREE.BufferGeometry().setFromPoints(points(outer))
    return {geometry,outline,kind}
  }) ?? [], [network])
  useEffect(() => () => meshes.forEach(({geometry,outline}) => {
    geometry.dispose()
    outline.dispose()
  }), [meshes])
  return <group rotation={[-Math.PI / 2,0,0]} position={[0,.023,0]}>
    {meshes.map(({geometry,outline,kind},index) => <group key={index} position={[0,0,kind === 'water' ? .006 : 0]}>
      <mesh geometry={geometry} receiveShadow><meshStandardMaterial color={colors[kind]} roughness={1}/></mesh>
      {kind !== 'water' && <lineLoop geometry={outline} position={[0,0,.003]}><lineBasicMaterial color={boundary}/></lineLoop>}
    </group>)}
  </group>
}
