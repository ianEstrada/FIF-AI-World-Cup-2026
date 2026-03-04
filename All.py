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
    num_poses=4)
detector_cuerpo = PoseLandmarker.create_from_options(opciones_pose)

opciones_manos = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=4)
detector_manos = mp.tasks.vision.HandLandmarker.create_from_options(opciones_manos)

INFO_EQUIPACIONES = {1: "Argentina", 2: "Portugal"}
TRAJES_DISPONIBLES = {
    1: cv2.imread('Body/argentina.png', cv2.IMREAD_UNCHANGED),
    2: cv2.imread('Body/portugal.png', cv2.IMREAD_UNCHANGED),
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
    return dedos

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

        lista_manos = []
        if res_hands_cache and res_hands_cache.hand_landmarks:
            for i, hand_lms in enumerate(res_hands_cache.hand_landmarks):
                num_dedos = contar_dedos(hand_lms, res_hands_cache.handedness[i][0].category_name)
                lista_manos.append({'dedos': num_dedos, 'x': hand_lms[0].x})

        if res_pose_cache and res_pose_cache.pose_landmarks:
            for i, pose_lms in enumerate(res_pose_cache.pose_landmarks):
                # Landmarks escalados al tamaño original del frame
                nariz = (int(pose_lms[0].x * w), int(pose_lms[0].y * h))
                px_h_izq = (int(pose_lms[11].x * w), int(pose_lms[11].y * h))
                px_h_der = (int(pose_lms[12].x * w), int(pose_lms[12].y * h))
                
                dist_hombros = abs(px_h_izq[0] - px_h_der[0])
                centro_x_rel = (pose_lms[11].x + pose_lms[12].x) / 2
                
                for mano in lista_manos:
                    if abs(mano['x'] - centro_x_rel) < 0.35:
                        if mano['dedos'] in TRAJES_DISPONIBLES:
                            trajes_activos[i] = mano['dedos']
                        break

                traje_id = trajes_activos.get(i)
                if traje_id:
                    angulo = math.degrees(math.atan2(px_h_izq[1] - px_h_der[1], px_h_izq[0] - px_h_der[0]))
                    ancho_t = dist_hombros * 1.8
                    
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

        # ==================== INTERFAZ (Tu diseño original) ====================
        h_f, w_f = frame.shape[:2]
        cv2.rectangle(frame, (15, 15), (320, 90), (20, 20, 40), -1)
        cv2.rectangle(frame, (15, 15), (320, 90), (0, 200, 255), 3)
        cv2.putText(frame, "WORLD CUP 2026", (30, 45), 2, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "VIRTUAL TRY-ON", (30, 70), 1, 0.7, (0, 200, 255), 2)
        
        cv2.rectangle(frame, (w_f - 350, 15), (w_f - 15, 110), (20, 20, 40), -1)
        cv2.rectangle(frame, (w_f - 350, 15), (w_f - 15, 110), (255, 255, 255), 2)
        cv2.putText(frame, "Selecciona tu equipo:", (w_f - 335, 40), 1, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "1 - ARGENTINA", (w_f - 335, 65), 2, 0.7, (100, 150, 255), 2)
        cv2.putText(frame, "2 - PORTUGAL", (w_f - 335, 90), 2, 0.7, (0, 200, 100), 2)
        
        cv2.rectangle(frame, (0, h_f - 50), (w_f, h_f), (20, 20, 40), -1)
        cv2.line(frame, (0, h_f - 50), (w_f, h_f - 50), (0, 200, 255), 3)
        
        cv2.imshow('AI Swapper - WORLD CUP 2026 EDITION', frame)
        t_start = time.time()
        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()