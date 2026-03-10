import cv2
import numpy as np
import math
import mediapipe as mp
import time
import cupy as cp

# ==========================================
# 1. CONFIGURACIÓN DE ALTO RENDIMIENTO
# ==========================================
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Mantenemos LITE para fluidez extrema
opciones_pose = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=8)
detector_cuerpo = PoseLandmarker.create_from_options(opciones_pose)

opciones_manos = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=8)
detector_manos = mp.tasks.vision.HandLandmarker.create_from_options(opciones_manos)

INFO_EQUIPACIONES = {
    1: "Argentina", 2: "Portugal", 3: "Mexico", 4: "España",
    5: "Inglaterra", 6: "Francia", 7: "Italia", 8: "USA",
    9: "Alemania", 10: "Colombia"
}
TRAJES_DISPONIBLES = {
    1: cv2.imread('Body/argentina.png', cv2.IMREAD_UNCHANGED),
    2: cv2.imread('Body/portugal.png', cv2.IMREAD_UNCHANGED),
    3: cv2.imread('Body/mexico.png', cv2.IMREAD_UNCHANGED),
    4: cv2.imread('Body/espana.png', cv2.IMREAD_UNCHANGED),
    5: cv2.imread('Body/inglaterra.png', cv2.IMREAD_UNCHANGED),
    6: cv2.imread('Body/francia.png', cv2.IMREAD_UNCHANGED),
    7: cv2.imread('Body/italia.png', cv2.IMREAD_UNCHANGED),
    8: cv2.imread('Body/usa.png', cv2.IMREAD_UNCHANGED),
    9: cv2.imread('Body/alemania.png', cv2.IMREAD_UNCHANGED),
    10:cv2.imread('Body/colombia.png', cv2.IMREAD_UNCHANGED),
}

# Suavizado para evitar jittering
suavizado_traje = {}
suavizado_hud = {}

# Variables para Skip Frames
res_pose_cache = None
res_hands_cache = None

# ==========================================
# 2. FUNCIONES OPTIMIZADAS
# ==========================================

def superponer_traje_fast(fondo, traje_png, x, y, ancho, alto):
    """Renderizado acelerado por RTX 2080 mediante CuPy"""
    try:
        ancho, alto = int(ancho), int(alto)
        if ancho <= 0 or alto <= 0: return fondo
        
        traje_res = cv2.resize(traje_png, (ancho, alto), interpolation=cv2.INTER_LINEAR)
        
        y1, y2 = max(0, y), min(fondo.shape[0], y + alto)
        x1, x2 = max(0, x), min(fondo.shape[1], x + ancho)
        t_y1, t_y2 = max(0, -y), min(alto, fondo.shape[0] - y)
        t_x1, t_x2 = max(0, -x), min(ancho, fondo.shape[1] - x)

        if y1 >= y2 or x1 >= x2: return fondo

        # Transferencia a VRAM (GPU)
        region_fondo_gpu = cp.asarray(fondo[y1:y2, x1:x2])
        region_traje_gpu = cp.asarray(traje_res[t_y1:t_y2, t_x1:t_x2])

        alpha = (region_traje_gpu[:, :, 3] / 255.0)[:, :, cp.newaxis]
        img_rgb = region_traje_gpu[:, :, :3]

        # Operación CUDA
        resultado_gpu = (alpha * img_rgb + (1 - alpha) * region_fondo_gpu).astype(cp.uint8)

        # Retorno a RAM (CPU)
        fondo[y1:y2, x1:x2] = cp.asnumpy(resultado_gpu)
        return fondo
    except:
        return fondo

def dibujar_hud_persona(frame, nariz_pos, hombros_dist, traje_id, persona_id):
    nombre = INFO_EQUIPACIONES.get(traje_id, "Seleccionando...")
    texto = f"SELECCION: {nombre} - 2026"
    
    key = f"hud_{persona_id}"
    if key not in suavizado_hud:
        suavizado_hud[key] = {'y': nariz_pos[1], 'x': nariz_pos[0]}
    
    target_y = nariz_pos[1] - int(hombros_dist * 0.3)
    target_x = nariz_pos[0]
    
    suavizado_hud[key]['y'] = int(suavizado_hud[key]['y'] * 0.7 + target_y * 0.3)
    suavizado_hud[key]['x'] = int(suavizado_hud[key]['x'] * 0.7 + target_x * 0.3)
    
    x, y = suavizado_hud[key]['x'], suavizado_hud[key]['y']
    font = cv2.FONT_HERSHEY_DUPLEX
    (w_t, h_t), _ = cv2.getTextSize(texto, font, 0.7, 2)
    
    cv2.rectangle(frame, (x - w_t//2 - 15, y - h_t - 20), (x + w_t//2 + 15, y + 10), (0, 0, 180), -1)
    cv2.rectangle(frame, (x - w_t//2 - 15, y - h_t - 20), (x + w_t//2 + 15, y + 10), (255, 255, 255), 2)
    cv2.putText(frame, texto, (x - w_t//2, y - 5), font, 0.7, (255, 255, 255), 2)

def rotar_imagen(imagen, angulo):
    h, w = imagen.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angulo, 1.0)
    return cv2.warpAffine(imagen, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

def dibujar_contorno_mano_glow(frame, hand_landmarks, ancho_frame, alto_frame):
    if not hand_landmarks: return
    puntos = [(int(lm.x * ancho_frame), int(lm.y * alto_frame)) for lm in hand_landmarks]
    conexiones = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
    for i, j in conexiones:
        cv2.line(frame, puntos[i], puntos[j], (0, 255, 255), 8)
        cv2.line(frame, puntos[i], puntos[j], (255, 255, 255), 2)
    for idx in [4, 8, 12, 16, 20]:
        cv2.circle(frame, puntos[idx], 6, (0, 255, 0), -1)

def contar_dedos(hand_landmarks, handedness):
    dedos = 0
    puntas, bases = [8, 12, 16, 20], [6, 10, 14, 18]
    for p, b in zip(puntas, bases):
        if hand_landmarks[p].y < hand_landmarks[b].y: dedos += 1
    if (handedness == "Left" and hand_landmarks[4].x < hand_landmarks[3].x) or \
       (handedness == "Right" and hand_landmarks[4].x > hand_landmarks[3].x):
        dedos += 1
    muneca = (hand_landmarks[0].x, hand_landmarks[0].y)
    return dedos, muneca


def obtener_dedos_por_mano(res_hands_cache):
    """Retorna diccionario con dedos por cada mano detectada, agrupadas por posición"""
    manos_dict = {}
    if res_hands_cache and res_hands_cache.hand_landmarks:
        for i, hand_lms in enumerate(res_hands_cache.hand_landmarks):
            handedness = res_hands_cache.handedness[i][0].category_name
            num_dedos, muneca = contar_dedos(hand_lms, handedness)
            clave = f"{handedness}_{i}"
            manos_dict[clave] = {'dedos': num_dedos, 'x': muneca[0], 'y': muneca[1], 'handedness': handedness}
    return manos_dict

# ==========================================
# 3. BUCLE PRINCIPAL
# ==========================================
def main():
    global res_pose_cache, res_hands_cache
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Pantalla Completa
    cv2.namedWindow('AI Swapper - WORLD CUP 2026 EDITION', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('AI Swapper - WORLD CUP 2026 EDITION', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    trajes_activos = {}
    ultimo_cambio_equipacion = 0
    COOLDOWN_CAMBIO = 1000  # ms
    frame_idx = 0
    t_start = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        timestamp = int(time.time() * 1000)

        # MEJORA: Inferencia en resolución reducida (IA vuela)
        frame_small = cv2.resize(frame, (640, 360))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))

        # MEJORA: Skip Frames (IA cada 2 cuadros)
        if frame_idx % 2 == 0:
            res_pose_cache = detector_cuerpo.detect_for_video(mp_image, timestamp)
            res_hands_cache = detector_manos.detect_for_video(mp_image, timestamp)
        
        frame_idx += 1

        # Obtener dedos de cada mano por separado (Left/Right)
        manos_por_lado = {}
        if res_hands_cache and res_hands_cache.hand_landmarks:
            for i, hand_lms in enumerate(res_hands_cache.hand_landmarks):
                handedness = res_hands_cache.handedness[i][0].category_name
                num_dedos, muneca = contar_dedos(hand_lms, handedness)
                manos_por_lado[handedness] = {'dedos': num_dedos, 'x': muneca[0], 'y': muneca[1]}
        
        # Calcular total de dedos (suma de ambas manos)
        dedos_izq = manos_por_lado.get('Left', {}).get('dedos', 0)
        dedos_der = manos_por_lado.get('Right', {}).get('dedos', 0)
        total_dedos = dedos_izq + dedos_der
        
        # Obtener posición promedio de las manos para asociación
        lista_manos = []
        if dedos_izq > 0:
            lista_manos.append({'dedos': dedos_izq, 'x': manos_por_lado['Left']['x'], 'y': manos_por_lado['Left']['y']})
        if dedos_der > 0:
            lista_manos.append({'dedos': dedos_der, 'x': manos_por_lado['Right']['x'], 'y': manos_por_lado['Right']['y']})
        
        # Si hay 2 manos, usar la suma total para el equipo
        usar_suma_dedos = (dedos_izq > 0 and dedos_der > 0)

        if res_pose_cache and res_pose_cache.pose_landmarks:
            for i, pose_lms in enumerate(res_pose_cache.pose_landmarks):
                nariz = (int(pose_lms[0].x * w), int(pose_lms[0].y * h))
                px_h_izq = (int(pose_lms[11].x * w), int(pose_lms[11].y * h))
                px_h_der = (int(pose_lms[12].x * w), int(pose_lms[12].y * h))
                
                dist_hombros = abs(px_h_izq[0] - px_h_der[0])
                centro_x_rel = (pose_lms[11].x + pose_lms[12].x) / 2
                
                # Determinar qué número de dedos usar
                nuevo_traje_id = None
                if usar_suma_dedos and total_dedos in TRAJES_DISPONIBLES:
                    nuevo_traje_id = total_dedos
                else:
                    for mano in lista_manos:
                        if abs(mano['x'] - centro_x_rel) < 0.35:
                            if mano['dedos'] in TRAJES_DISPONIBLES:
                                nuevo_traje_id = mano['dedos']
                            break
                
                # Aplicar cooldown para evitar cambios por micro-movimientos
                traje_actual = trajes_activos.get(i)
                if nuevo_traje_id is not None:
                    tiempo_desde_cambio = timestamp - ultimo_cambio_equipacion
                    if nuevo_traje_id != traje_actual and tiempo_desde_cambio >= COOLDOWN_CAMBIO:
                        trajes_activos[i] = nuevo_traje_id
                        ultimo_cambio_equipacion = timestamp

                traje_id = trajes_activos.get(i)
                if traje_id:
                    angulo = math.degrees(math.atan2(px_h_izq[1] - px_h_der[1], px_h_izq[0] - px_h_der[0]))
                    ancho_t = dist_hombros * 1.7
                    
                    key = f"traje_{i}"
                    if key not in suavizado_traje:
                        suavizado_traje[key] = {'x': (px_h_izq[0]+px_h_der[0])/2, 'y': px_h_izq[1], 'ancho': ancho_t, 'angulo': angulo}
                    
                    suavizado_traje[key]['x'] = suavizado_traje[key]['x'] * 0.6 + ((px_h_izq[0]+px_h_der[0])/2) * 0.4
                    suavizado_traje[key]['y'] = suavizado_traje[key]['y'] * 0.6 + min(px_h_izq[1], px_h_der[1]) * 0.4
                    suavizado_traje[key]['ancho'] = suavizado_traje[key]['ancho'] * 0.7 + ancho_t * 0.3
                    suavizado_traje[key]['angulo'] = suavizado_traje[key]['angulo'] * 0.7 + angulo * 0.3
                    
                    img_t = rotar_imagen(TRAJES_DISPONIBLES[traje_id], -suavizado_traje[key]['angulo'])
                    frame = superponer_traje_fast(frame, img_t, 
                        int(suavizado_traje[key]['x'] - suavizado_traje[key]['ancho']/2), 
                        int(suavizado_traje[key]['y'] - suavizado_traje[key]['ancho']*1.3*0.2), 
                        suavizado_traje[key]['ancho'], suavizado_traje[key]['ancho'] * 1.3)
                    
                    dibujar_hud_persona(frame, nariz, dist_hombros, traje_id, i)

        if res_hands_cache and res_hands_cache.hand_landmarks:
            for hand_lms in res_hands_cache.hand_landmarks:
                dibujar_contorno_mano_glow(frame, hand_lms, w, h)

        # ==================== INTERFAZ CON GRID 2x5 ====================
        h_f, w_f = frame.shape[:2]
        cv2.rectangle(frame, (15, 15), (320, 90), (20, 20, 40), -1)
        cv2.rectangle(frame, (15, 15), (320, 90), (0, 200, 255), 3)
        cv2.putText(frame, "WORLD CUP 2026", (30, 45), 2, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "VIRTUAL TRY-ON", (30, 70), 1, 0.7, (0, 200, 255), 2)
        
        # Panel derecho: Grid 2x5 organizado
        panel_x = w_f - 280
        panel_w = 265
        panel_h = 180
        cv2.rectangle(frame, (panel_x, 15), (panel_x + panel_w, panel_h + 15), (20, 20, 40), -1)
        cv2.rectangle(frame, (panel_x, 15), (panel_x + panel_w, panel_h + 15), (255, 255, 255), 2)
        cv2.putText(frame, "DEDOS = EQUIPO", (panel_x + 15, 40), 1, 0.7, (200, 200, 200), 1)
        
        equipos_grid = [
            (1, "Argentina", (100, 150, 255)),
            (2, "Portugal", (0, 200, 100)),
            (3, "Mexico", (255, 100, 100)),
            (4, "España", (255, 255, 100)),
            (5, "Inglaterra", (100, 255, 150)),
            (6, "Francia", (150, 255, 100)),
            (7, "Italia", (255, 150, 100)),
            (8, "USA", (100, 255, 255)),
            (9, "Alemania", (255, 100, 255)),
            (10, "Colombia", (255, 255, 150)),
        ]
        
        col_width = panel_w // 2
        row_height = 28
        for idx, (num, nombre, color) in enumerate(equipos_grid):
            col = idx % 2
            row = idx // 2
            x_pos = panel_x + 15 + col * col_width
            y_pos = 60 + row * row_height
            cv2.rectangle(frame, (x_pos, y_pos - 18), (x_pos + col_width - 20, y_pos + 2), color, -1)
            cv2.rectangle(frame, (x_pos, y_pos - 18), (x_pos + col_width - 20, y_pos + 2), (255, 255, 255), 1)
            cv2.putText(frame, f"{num}", (x_pos + 5, y_pos), 2, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, nombre, (x_pos + 25, y_pos), 1, 0.5, (255, 255, 255), 1)
        
        # Barra inferior con indicador visual
        cv2.rectangle(frame, (0, h_f - 60), (w_f, h_f), (20, 20, 40), -1)
        cv2.line(frame, (0, h_f - 60), (w_f, h_f - 60), (0, 200, 255), 3)
        
        # Indicador de dedos activos (mostrar dedos individuales y suma total)
        dedos_individual = []
        if dedos_izq > 0:
            dedos_individual.append(dedos_izq)
        if dedos_der > 0:
            dedos_individual.append(dedos_der)
        
        for d in range(1, 11):
            x_dedo = 30 + (d - 1) * ((w_f - 60) // 10)
            # Resaltar si es el número de dedos individual O la suma total
            es_seleccionado = (d == total_dedos) or (d in dedos_individual)
            color_dedo = (0, 255, 0) if es_seleccionado else (80, 80, 80)
            cv2.circle(frame, (x_dedo, h_f - 30), 15, color_dedo, -1)
            cv2.circle(frame, (x_dedo, h_f - 30), 15, (255, 255, 255), 2)
            cv2.putText(frame, str(d), (x_dedo - 5, h_f - 25), 1, 0.6, (0, 0, 0), 2)
        
        cv2.imshow('AI Swapper - WORLD CUP 2026 EDITION', frame)
        t_start = time.time()
        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()