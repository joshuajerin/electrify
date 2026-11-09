import { createRoot } from 'react-dom/client'
import App from './App'
import MissionV2 from './MissionV2'
import './style.css'

const Screen = window.location.pathname === '/legacy' ? App : MissionV2
createRoot(document.getElementById('root')!).render(<Screen/>)
