import cv2
import numpy as np
import math
import mediapipe as mp
import time

USE_GPU = False
cp = None
try:
    import cupy as cp
    USE_GPU = True
except ImportError:
    pass

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

# ==========================================
# 2. CONFIGURACIÓN DE JERSEYS
# ==========================================
INFO_EQUIPACIONES = {1: "Argentina", 2: "Portugal", 3: "Brasil", 4: "Francia"}
TRAJES_DISPONIBLES = {
    1: cv2.imread('Body/argentina.png', cv2.IMREAD_UNCHANGED),
    2: cv2.imread('Body/portugal.png', cv2.IMREAD_UNCHANGED),
}

# ==========================================
# 3. BUFFERS DE SMOOTHING TEMPORAL
# ==========================================
BUFFER_SIZE = 3
buffer_pose = {}
buffer_manos = {}

# Confianza mínima para aceptar detecciones
MIN_CONFIANZA = 0.5

# ==========================================
# 4. FUNCIONES OPTIMIZADAS
# ==========================================

def distancia(p1, p2):
    """Calcula distancia euclidiana entre dos puntos."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def obtener_landmarks_promediados(buffer, key):
    """Promedia los últimos frames del buffer para smoothing."""
    if key not in buffer or len(buffer[key]) == 0:
        return None
    
    frames = buffer[key]
    num_frames = len(frames)
    
    resultado = []
    for i in range(len(frames[0])):
        sum_x = sum(frame[i].x for frame in frames)
        sum_y = sum(frame[i].y for frame in frames)
        resultado.append(type('Landmark', (), {
            'x': sum_x / num_frames,
            'y': sum_y / num_frames,
            'presence': sum(getattr(frame[i], 'presence', 1) for frame in frames) / num_frames
        })())
    
    return resultado


def agregar_al_buffer(buffer, key, datos):
    """Agrega datos al buffer circular."""
    if key not in buffer:
        buffer[key] = []
    buffer[key].append(datos)
    if len(buffer[key]) > BUFFER_SIZE:
        buffer[key].pop(0)


def superponer_traje_fast(fondo, traje_png, x, y, ancho, alto):
    """Renderizado con fallback CPU/GPU."""
    try:
        ancho, alto = int(ancho), int(alto)
        if ancho <= 0 or alto <= 0:
            return fondo
        
        traje_res = cv2.resize(traje_png, (ancho, alto), interpolation=cv2.INTER_LINEAR)
        
        y1, y2 = max(0, y), min(fondo.shape[0], y + alto)
        x1, x2 = max(0, x), min(fondo.shape[1], x + ancho)
        t_y1, t_y2 = max(0, -y), min(alto, fondo.shape[0] - y)
        t_x1, t_x2 = max(0, -x), min(ancho, fondo.shape[1] - x)

        if y1 >= y2 or x1 >= x2:
            return fondo

        if USE_GPU and cp is not None:
            region_fondo_gpu = cp.asarray(fondo[y1:y2, x1:x2])
            region_traje_gpu = cp.asarray(traje_res[t_y1:t_y2, t_x1:t_x2])
            alpha = (region_traje_gpu[:, :, 3] / 255.0)[:, :, cp.newaxis]
            img_rgb = region_traje_gpu[:, :, :3]
            resultado_gpu = (alpha * img_rgb + (1 - alpha) * region_fondo_gpu).astype(cp.uint8)
            fondo[y1:y2, x1:x2] = cp.asnumpy(resultado_gpu)
        else:
            alpha = traje_res[t_y1:t_y2, t_x1:t_x2, 3] / 255.0
            alpha_inv = 1.0 - alpha
            for c in range(3):
                fondo[y1:y2, x1:x2, c] = (
                    alpha * traje_res[t_y1:t_y2, t_x1:t_x2, c] +
                    alpha_inv * fondo[y1:y2, x1:x2, c]
                ).astype(np.uint8)
        
        return fondo
    except Exception as e:
        print(f"Error en superponer_traje: {e}")
        return fondo


def dibujar_hud_persona(frame, nariz_pos, hombros_dist, traje_id, persona_id):
    """Dibuja el HUD sobre la persona."""
    nombre = INFO_EQUIPACIONES.get(traje_id, "Seleccionando...")
    texto = f"SELECCION: {nombre} - 2026"
    
    key = f"hud_{persona_id}"
    
    target_y = nariz_pos[1] - int(hombros_dist * 0.3)
    target_x = nariz_pos[0]
    
    if key not in suavizado_hud:
        suavizado_hud[key] = {'y': target_y, 'x': target_x}
    
    suavizado_hud[key]['y'] = int(suavizado_hud[key]['y'] * 0.7 + target_y * 0.3)
    suavizado_hud[key]['x'] = int(suavizado_hud[key]['x'] * 0.7 + target_x * 0.3)
    
    x, y = suavizado_hud[key]['x'], suavizado_hud[key]['y']
    font = cv2.FONT_HERSHEY_DUPLEX
    (w_t, h_t), _ = cv2.getTextSize(texto, font, 0.7, 2)
    
    cv2.rectangle(frame, (x - w_t//2 - 15, y - h_t - 20), (x + w_t//2 + 15, y + 10), (0, 0, 180), -1)
    cv2.rectangle(frame, (x - w_t//2 - 15, y - h_t - 20), (x + w_t//2 + 15, y + 10), (255, 255, 255), 2)
    cv2.putText(frame, texto, (x - w_t//2, y - 5), font, 0.7, (255, 255, 255), 2)


def rotar_imagen(imagen, angulo):
    """Rota una imagen alrededor de su centro."""
    if imagen is None or imagen.size == 0:
        return imagen
    h, w = imagen.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angulo, 1.0)
    return cv2.warpAffine(imagen, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))


def dibujar_contorno_mano_glow(frame, hand_landmarks, ancho_frame, alto_frame):
    """Dibuja el contorno de la mano con efecto glow."""
    if not hand_landmarks:
        return
    puntos = [(int(lm.x * ancho_frame), int(lm.y * alto_frame)) for lm in hand_landmarks]
    conexiones = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
    for i, j in conexiones:
        cv2.line(frame, puntos[i], puntos[j], (0, 255, 255), 8)
        cv2.line(frame, puntos[i], puntos[j], (255, 255, 255), 2)
    for idx in [4, 8, 12, 16, 20]:
        cv2.circle(frame, puntos[idx], 6, (0, 255, 0), -1)


def contar_dedos(hand_landmarks, handedness):
    """Cuenta el número de dedos levantados."""
    if not hand_landmarks:
        return 0
    dedos = 0
    puntas, bases = [8, 12, 16, 20], [6, 10, 14, 18]
    for p, b in zip(puntas, bases):
        if hand_landmarks[p].y < hand_landmarks[b].y:
            dedos += 1
    if (handedness == "Left" and hand_landmarks[4].x < hand_landmarks[3].x) or \
       (handedness == "Right" and hand_landmarks[4].x > hand_landmarks[3].x):
        dedos += 1
    return dedos


def verificar_confianza_landmarks(pose_lms):
    """Verifica que los landmarks principales tengan confianza suficiente."""
    landmarks_importantes = [11, 12, 23, 24]  # hombros y caderas
    for idx in landmarks_importantes:
        if idx >= len(pose_lms):
            return False
        presence = getattr(pose_lms[idx], 'presence', 1.0)
        if presence < MIN_CONFIANZA:
            return False
    return True


# Suavizado para HUD
suavizado_traje = {}
suavizado_hud = {}

# ==========================================
# 5. BUCLE PRINCIPAL
# ==========================================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    cv2.namedWindow('AI Swapper - WORLD CUP 2026 EDITION', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('AI Swapper - WORLD CUP 2026 EDITION', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(f"Iniciando AI Swapper (GPU: {'Si' if USE_GPU else 'No'})...")

    trajes_activos = {}
    frame_idx = 0
    fps_loop = 0
    fps_start = time.time()
    fps_display = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        timestamp = int(time.time() * 1000)

        # Inferencia en resolución nativa (720p)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Ejecutar IA cada frame (sin skip)
        res_pose_cache = detector_cuerpo.detect_for_video(mp_image, timestamp)
        res_hands_cache = detector_manos.detect_for_video(mp_image, timestamp)

        # Procesar manos
        lista_manos = []
        if res_hands_cache and res_hands_cache.hand_landmarks:
            for i, hand_lms in enumerate(res_hands_cache.hand_landmarks):
                handedness = res_hands_cache.handedness[i][0].category_name if res_hands_cache.handedness else "Right"
                num_dedos = contar_dedos(hand_lms, handedness)
                
                # Posición de la muñeca (landmark 0)
                muneca_pos = (int(hand_lms[0].x * w), int(hand_lms[0].y * h))
                
                lista_manos.append({
                    'dedos': num_dedos,
                    'x': hand_lms[0].x,
                    'y': hand_lms[0].y,
                    'pos': muneca_pos,
                    'handedness': handedness
                })

        # Procesar cuerpos
        if res_pose_cache and res_pose_cache.pose_landmarks:
            for i, pose_lms in enumerate(res_pose_cache.pose_landmarks):
                # Filtrar por confianza
                if not verificar_confianza_landmarks(pose_lms):
                    continue
                
                # Agregar al buffer de smoothing
                agregar_al_buffer(buffer_pose, i, pose_lms)
                
                # Obtener landmarks promediados
                pose_lms_smooth = obtener_landmarks_promediados(buffer_pose, i)
                if not pose_lms_smooth:
                    continue
                
                # Landmarks escalados al tamaño original del frame
                nariz = (int(pose_lms_smooth[0].x * w), int(pose_lms_smooth[0].y * h))
                
                # Hombros (11, 12)
                px_h_izq = (int(pose_lms_smooth[11].x * w), int(pose_lms_smooth[11].y * h))
                px_h_der = (int(pose_lms_smooth[12].x * w), int(pose_lms_smooth[12].y * h))
                
                # Caderas (23, 24) - nuevos landmarks
                px_cadera_izq = (int(pose_lms_smooth[23].x * w), int(pose_lms_smooth[23].y * h))
                px_cadera_der = (int(pose_lms_smooth[24].x * w), int(pose_lms_smooth[24].y * h))
                
                # Codos (13, 14) - nuevos landmarks
                px_codo_izq = (int(pose_lms_smooth[13].x * w), int(pose_lms_smooth[13].y * h))
                px_codo_der = (int(pose_lms_smooth[14].x * w), int(pose_lms_smooth[14].y * h))
                
                # Calcular dimensiones
                dist_hombros = abs(px_h_izq[0] - px_h_der[0])
                dist_caderas = abs(px_cadera_izq[0] - px_cadera_der[0])
                dist_codos = abs(px_codo_izq[0] - px_codo_der[0])
                
                # Usar el mayor ancho detectado (codos o hombros)
                ancho_torso = max(dist_hombros, dist_codos)
                
                centro_x_rel = (pose_lms_smooth[11].x + pose_lms_smooth[12].x) / 2
                centro_x = int(centro_x_rel * w)
                
                # Mejor asociación mano-cuerpo usando distancia euclidiana
                mejor_mano = None
                mejor_distancia = float('inf')
                
                for mano in lista_manos:
                    # Distancia a ambos hombros
                    dist_h_izq = distancia(mano['pos'], px_h_izq)
                    dist_h_der = distancia(mano['pos'], px_h_der)
                    dist_min = min(dist_h_izq, dist_h_der)
                    
                    # También verificar que esté en el lado correcto del cuerpo
                    lado_correcto = (
                        (mano['x'] < centro_x_rel + 0.1 and mano['handedness'] == "Left") or
                        (mano['x'] > centro_x_rel - 0.1 and mano['handedness'] == "Right")
                    )
                    
                    if dist_min < mejor_distancia and lado_correcto:
                        mejor_distancia = dist_min
                        mejor_mano = mano
                
                # Threshold para asociar mano (en pixels, ajustado por escala)
                threshold_asociacion = ancho_torso * 0.8
                
                if mejor_mano and mejor_distancia < threshold_asociacion:
                    if mejor_mano['dedos'] in TRAJES_DISPONIBLES:
                        trajes_activos[i] = mejor_mano['dedos']
                else:
                    if i in trajes_activos:
                        del trajes_activos[i]

                # Renderizar jersey si hay selección activa
                traje_id = trajes_activos.get(i)
                if traje_id and traje_id in TRAJES_DISPONIBLES:
                    # Calcular ángulo de hombros
                    angulo = math.degrees(math.atan2(px_h_izq[1] - px_h_der[1], px_h_izq[0] - px_h_der[0]))
                    
                    # Calcular ancho del jersey (usando caderas para mejor ajuste)
                    ancho_t = ancho_torso * 2.0
                    alto_t = ancho_t * 1.3
                    
                    key = f"traje_{i}"
                    if key not in suavizado_traje:
                        suavizado_traje[key] = {
                            'x': centro_x,
                            'y': min(px_h_izq[1], px_h_der[1]),
                            'ancho': ancho_t,
                            'alto': alto_t,
                            'angulo': angulo
                        }
                    
                    # Smoothing
                    suavizado_traje[key]['x'] = int(suavizado_traje[key]['x'] * 0.7 + centro_x * 0.3)
                    suavizado_traje[key]['y'] = int(suavizado_traje[key]['y'] * 0.7 + min(px_h_izq[1], px_h_der[1]) * 0.3)
                    suavizado_traje[key]['ancho'] = int(suavizado_traje[key]['ancho'] * 0.7 + ancho_t * 0.3)
                    suavizado_traje[key]['alto'] = int(suavizado_traje[key]['alto'] * 0.7 + alto_t * 0.3)
                    suavizado_traje[key]['angulo'] = suavizado_traje[key]['angulo'] * 0.7 + angulo * 0.3
                    
                    img_t = rotar_imagen(TRAJES_DISPONIBLES[traje_id], -suavizado_traje[key]['angulo'])
                    
                    if img_t is not None:
                        frame = superponer_traje_fast(
                            frame, img_t,
                            int(suavizado_traje[key]['x'] - suavizado_traje[key]['ancho'] // 2),
                            int(suavizado_traje[key]['y'] - suavizado_traje[key]['alto'] * 0.15),
                            suavizado_traje[key]['ancho'],
                            suavizado_traje[key]['alto']
                        )
                    
                    dibujar_hud_persona(frame, nariz, ancho_torso, traje_id, i)

        # Dibujar manos
        if res_hands_cache and res_hands_cache.hand_landmarks:
            for hand_lms in res_hands_cache.hand_landmarks:
                dibujar_contorno_mano_glow(frame, hand_lms, w, h)

        # FPS counter
        fps_loop += 1
        if time.time() - fps_start >= 1.0:
            fps_display = fps_loop
            fps_loop = 0
            fps_start = time.time()

        # Interfaz
        h_f, w_f = frame.shape[:2]
        
        # Panel izquierdo - Título
        cv2.rectangle(frame, (15, 15), (320, 90), (20, 20, 40), -1)
        cv2.rectangle(frame, (15, 15), (320, 90), (0, 200, 255), 3)
        cv2.putText(frame, "WORLD CUP 2026", (30, 45), 2, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "VIRTUAL TRY-ON", (30, 70), 1, 0.7, (0, 200, 255), 2)
        
        # Panel derecho - Opciones
        cv2.rectangle(frame, (w_f - 350, 15), (w_f - 15, 130), (20, 20, 40), -1)
        cv2.rectangle(frame, (w_f - 350, 15), (w_f - 15, 130), (255, 255, 255), 2)
        cv2.putText(frame, "Selecciona tu equipo:", (w_f - 335, 40), 1, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "1 - ARGENTINA", (w_f - 335, 65), 2, 0.7, (100, 150, 255), 2)
        cv2.putText(frame, "2 - PORTUGAL", (w_f - 335, 90), 2, 0.7, (0, 200, 100), 2)
        
        # FPS display
        cv2.putText(frame, f"FPS: {fps_display}", (w_f - 120, 120), 1, 0.6, (0, 255, 0), 1)
        
        # Barra inferior
        cv2.rectangle(frame, (0, h_f - 50), (w_f, h_f), (20, 20, 40), -1)
        cv2.line(frame, (0, h_f - 50), (w_f, h_f - 50), (0, 200, 255), 3)
        
        cv2.imshow('AI Swapper - WORLD CUP 2026 EDITION', frame)
        
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
