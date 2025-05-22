import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
from deepface import DeepFace

class PersonFinderAI:
    def __init__(self, target_photo_paths, face_threshold=0.4):
        """
        Initialize the PersonFinderAI with target photos and parameters
        
        Args:
            target_photo_paths: List of paths to target person's photos
            face_threshold: Threshold for face similarity (lower = more strict for cosine distance)
        """
        self.target_photo_paths = target_photo_paths if isinstance(target_photo_paths, list) else [target_photo_paths]
        self.face_threshold = face_threshold
        self.target_face_embeddings = []  # Lista de embeddings de todas las fotos objetivo
        self.detection_history = []  # Para mantener un historial de detecciones
        self.total_detections = 0
        self.history_size = 15  # Tamaño del historial para promediar detecciones
        
    def load_target_embeddings(self):
        """
        Extract face embeddings from all target photos using DeepFace
        """
        print("\nCargando fotos objetivo...")
        self.target_face_embeddings = []
        
        for photo_path in self.target_photo_paths:
            try:
                # Verificar si el archivo existe
                if not os.path.exists(photo_path):
                    print(f"❌ Error: El archivo no existe: {photo_path}")
                    continue
                
                # Verificar si la imagen se puede leer con OpenCV
                img = cv2.imread(photo_path)
                if img is None:
                    print(f"❌ Error: No se pudo leer la imagen con OpenCV: {photo_path}")
                    continue
                    
                print(f"📸 Procesando imagen: {os.path.basename(photo_path)}")
                print(f"   Dimensiones: {img.shape}")
                
                # Extract face embedding using DeepFace
                embedding_objs = DeepFace.represent(
                    img_path=photo_path,
                    model_name="Facenet",
                    detector_backend="retinaface"
                )
                
                if embedding_objs and len(embedding_objs) > 0:
                    self.target_face_embeddings.append(embedding_objs[0]["embedding"])
                    print(f"✅ Rostro objetivo extraído correctamente de: {os.path.basename(photo_path)}")
                else:
                    print(f"⚠️ Advertencia: No se detectó rostro en la foto objetivo: {os.path.basename(photo_path)}")
                    
            except Exception as e:
                print(f"❌ Error al cargar la foto objetivo {os.path.basename(photo_path)}: {str(e)}")
                print(f"   Ruta completa: {os.path.abspath(photo_path)}")
                print(f"   Tipo de error: {type(e).__name__}")
        
        if not self.target_face_embeddings:
            raise Exception("No se pudo extraer ningún embedding de las fotos objetivo. Verifica que las fotos existan y contengan rostros detectables.")
            
        print(f"✅ Se cargaron {len(self.target_face_embeddings)} embeddings de referencia")
        
    def process_frame(self, frame):
        """
        Process a single frame to detect the target person using DeepFace
        Returns frame with annotations
        """
        try:
            # Detect and extract embeddings from all faces in the frame
            frame_embeddings = DeepFace.represent(
                img_path=frame,
                model_name="Facenet",
                detector_backend="retinaface"
            )
            
            # Lista para almacenar todas las detecciones de este frame
            frame_detections = []
            
            for face_data in frame_embeddings:
                facial_area = face_data['facial_area']
                detection = {
                    'bbox': [facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']],
                    'confidence': face_data.get('confidence', 1.0)
                }
                
                # Calcular similitud con todos los rostros objetivo
                if self.target_face_embeddings:
                    face_embedding = face_data["embedding"]
                    # Calcular similitud con cada embedding objetivo y tomar el mejor resultado
                    best_similarity = float('inf')
                    for target_embedding in self.target_face_embeddings:
                        # Usar distancia coseno (menor = más similar)
                        face_sim = 1 - np.dot(face_embedding, target_embedding) / (
                            np.linalg.norm(face_embedding) * np.linalg.norm(target_embedding)
                        )
                        best_similarity = min(best_similarity, face_sim)
                    
                    if best_similarity < self.face_threshold:  # Menor distancia = mayor similitud
                        detection['match_type'] = 'Face'
                        detection['similarity'] = 1 - best_similarity  # Convertir distancia a similitud
                        self.total_detections += 1
                        self.detection_history.append(detection)
                        if len(self.detection_history) > self.history_size:
                            self.detection_history.pop(0)
                
                frame_detections.append(detection)
            
            # Dibujar todas las detecciones
            for detection in frame_detections:
                self._draw_detection(frame, detection)
                
        except Exception as e:
            print(f"⚠️ Error procesando frame: {str(e)}")
            
        return frame
    
    def _draw_detection(self, frame, detection):
        """
        Draw bounding box and labels on frame
        """
        x, y, w, h = detection['bbox']
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        
        # Color verde para detección facial, gris para personas no objetivo
        if 'match_type' in detection:
            # Es una coincidencia con la persona objetivo
            color = (0, 255, 0)  # Verde para detección facial
            # Calcular confianza promedio usando el historial
            recent_similarities = [d['similarity'] for d in self.detection_history[-10:] 
                                if d['match_type'] == detection['match_type']]
            avg_similarity = sum(recent_similarities) / len(recent_similarities) if recent_similarities else detection['similarity']
            confidence_pct = int(avg_similarity * 100)
            label = f"Persona Objetivo ({confidence_pct}%)"
        else:
            # Es una detección de persona no objetivo
            color = (128, 128, 128)  # Gris para otras personas
            label = "Persona"
        
        # Dibujar bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Dibujar fondo para el texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        label_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        cv2.rectangle(frame,
                     (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1),
                     color,
                     cv2.FILLED)
        
        # Dibujar texto
        cv2.putText(frame,
                    label,
                    (x1, y1 - 5),
                    font,
                    font_scale,
                    (255, 255, 255),  # Texto blanco
                    font_thickness)
    
    def process_video(self, input_video_path, output_video_path):
        """
        Process entire video and save output
        """
        # Load target embeddings first
        self.load_target_embeddings()
        
        print("\nProcesando video...")
        # Open video capture
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {input_video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = total_frames / fps
        
        print(f"\nInformación del video:")
        print(f"📊 Resolución: {width}x{height}")
        print(f"⏱️ FPS: {fps}")
        print(f"⌛ Duración: {duration:.1f} segundos")
        print(f"🎞️ Frames totales: {total_frames}")
        
        # Create video writer
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Initialize progress bar
        pbar = tqdm(total=total_frames, desc="Procesando frames", 
                   unit="frames", dynamic_ncols=True)
        
        start_time = time.time()
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Write frame
                out.write(processed_frame)
                
                # Update progress bar
                pbar.update(1)
                
                # Update progress bar description with detection stats
                if self.total_detections > 0:
                    pbar.set_postfix({
                        "Detecciones": self.total_detections
                    })
                
        finally:
            end_time = time.time()
            processing_time = end_time - start_time
            
            cap.release()
            out.release()
            pbar.close()
            
            print("\n✅ Procesamiento completado!")
            print(f"\nEstadísticas finales:")
            print(f"⏱️ Tiempo total: {processing_time:.1f} segundos")
            print(f"🎯 Detecciones totales: {self.total_detections}")
            print(f"📊 FPS promedio: {total_frames/processing_time:.1f}")
            print(f"\n💾 Video guardado en: {output_video_path}")

    def process_image(self, input_image_path, output_image_path):
        """
        Process a single image and save output
        """
        # Load target embeddings first
        self.load_target_embeddings()
        
        print("\nProcesando imagen...")
        print(f"Intentando abrir: {os.path.abspath(input_image_path)}")
        
        # Verify file exists
        if not os.path.exists(input_image_path):
            raise FileNotFoundError(f"El archivo no existe en la ruta: {input_image_path}")
        
        # Read the image
        frame = cv2.imread(input_image_path)
        if frame is None:
            raise FileNotFoundError(f"OpenCV no pudo leer la imagen: {input_image_path}. Verifica el formato del archivo.")
            
        # Process frame
        processed_frame = self.process_frame(frame)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        
        # Save the processed image
        cv2.imwrite(output_image_path, processed_frame)
        
        print("\n✅ Procesamiento completado!")
        print(f"🎯 Detecciones totales: {self.total_detections}")
        print(f"\n💾 Imagen guardada en: {output_image_path}")

def main():
    # Define paths using os.path for better compatibility
    target_photos = [
        os.path.join("photo_target", "AdamSandler", "1.jpg"),
        os.path.join("photo_target", "AdamSandler", "2.jpg"),
        os.path.join("photo_target", "AdamSandler", "3.jpg"),
        os.path.join("photo_target", "AdamSandler", "4.jpg"),
        os.path.join("photo_target", "AdamSandler", "5.jpg"),
        os.path.join("photo_target", "AdamSandler", "6.jpg"),
        os.path.join("photo_target", "AdamSandler", "7.jpg")            
    ]
    
    print("\n🔍 Person Finder AI - Iniciando...")
    
    # Create instance of PersonFinderAI
    finder = PersonFinderAI(target_photos, face_threshold=0.5)
    
    # Check if input is video or image
    input_path = os.path.join("input","Adam", "prueba6.mp4")
    is_video = input_path.lower().endswith(('.mp4', '.avi', '.mov', '.wmv'))
    
    if is_video:
        output_path = os.path.join("output", "Adam", "detections_output6.mp4")
        finder.process_video(input_path, output_path)
    else:
        output_path = os.path.join("output", "Adam", "detection_output4.jpg")
        finder.process_image(input_path, output_path)

if __name__ == "__main__":
    main() 