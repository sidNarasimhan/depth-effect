import cv2
import numpy as np
import pygame
import dlib
import os

def upload_image():
    image = cv2.imread("image.jpg", cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read image. Please check the path.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)
    print("Image min and max values:", image.min(), image.max())
    return image

def load_depth_map(image_shape):
    depth_map = cv2.imread("depth.png", cv2.IMREAD_GRAYSCALE)
    if depth_map is None:
        raise ValueError("Could not read depth map. Please check if 'depth.png' exists.")
    
    if depth_map.shape[:2] != image_shape[:2]:
        print("Depth map dimensions do not match the image. Resizing.")
        depth_map = cv2.resize(depth_map, (image_shape[1], image_shape[0]))
    
    depth_map = depth_map.astype(np.float32) / 255.0  # Normalize to 0-1
    print("Depth map loaded.")
    print("Depth map shape:", depth_map.shape)
    print("Depth map dtype:", depth_map.dtype)
    print("Depth map min and max values:", depth_map.min(), depth_map.max())
    return depth_map

def get_eye_position(frame, detector, predictor, prev_position=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) > 0:
        landmarks = predictor(gray, faces[0])
        left_eye = np.mean([(landmarks.part(36).x, landmarks.part(36).y),
                            (landmarks.part(39).x, landmarks.part(39).y)], axis=0)
        right_eye = np.mean([(landmarks.part(42).x, landmarks.part(42).y),
                             (landmarks.part(45).x, landmarks.part(45).y)], axis=0)
        current_position = np.mean([left_eye, right_eye], axis=0)
        
        if prev_position is not None:
            smoothed_position = prev_position * 0.6 + current_position * 0.4  # Increased sensitivity
            return smoothed_position
        else:
            return current_position
    return prev_position if prev_position is not None else np.array([frame.shape[1]/2, frame.shape[0]/2])

def get_mouse_position(screen_width, screen_height):
    x, y = pygame.mouse.get_pos()
    return np.array([x, y])

def apply_depth_effect(image, depth_map, position, max_offset=10):  # Reduced max_offset
    height, width = depth_map.shape
    y, x = np.mgrid[0:height, 0:width]
    
    # Calculate normalized position with reduced sensitivity
    norm_x = (position[0] - width / 2) / (width )  # Reduced sensitivity
    norm_y = (position[1] - height / 2) / (height )  # Reduced sensitivity
    
    # Invert depth map so lighter parts move more
    inv_depth_map = 1- depth_map
    
    # Apply offset based on normalized position and inverted depth
    offset_x = -norm_x * max_offset * inv_depth_map
    offset_y = -norm_y * max_offset * inv_depth_map
    
    x_map = x + offset_x
    y_map = y + offset_y
    
    # Clip values to prevent out-of-bounds access
    x_map = np.clip(x_map, 0, width - 1)
    y_map = np.clip(y_map, 0, height - 1)
    
    # Remap image
    remapped = cv2.remap(image, x_map.astype(np.float32), y_map.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    
    return remapped

def render_image(screen, image):
    # Zoom in by 20%
    zoom = 1
    h, w = image.shape[:2]
    zoomed_h, zoomed_w = int(h * zoom), int(w * zoom)
    
    # Crop the center of the zoomed image
    start_y = (zoomed_h - h) // 2
    start_x = (zoomed_w - w) // 2
    zoomed_image = cv2.resize(image, (zoomed_w, zoomed_h), interpolation=cv2.INTER_LANCZOS4)[start_y:start_y+h, start_x:start_x+w]
    
    # Convert the image to a pygame surface directly
    surface = pygame.surfarray.make_surface(zoomed_image.swapaxes(0, 1))
    screen.blit(surface, (0, 0))
    pygame.display.flip()

def main():
    image = upload_image()
    depth_map = load_depth_map(image.shape)
    
    pygame.init()
    screen = pygame.display.set_mode((image.shape[1], image.shape[0]))
    pygame.display.set_caption("Parallax Effect")
    clock = pygame.time.Clock()
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    cap = cv2.VideoCapture(0)
    
    running = True
    prev_eye_position = None
    use_mouse = True  # Flag to switch between eye tracking and mouse movement
    
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    use_mouse = not use_mouse
                    print("Using mouse:" if use_mouse else "Using eye tracking")
        
        if use_mouse:
            position = get_mouse_position(image.shape[1], image.shape[0])
        else:
            position = get_eye_position(frame, detector, predictor, prev_eye_position)
            prev_eye_position = position
        
        if position is not None:
            depth_affected_image = apply_depth_effect(image, depth_map, position, max_offset=20)  # Reduced max_offset
        else:
            depth_affected_image = image
        
        render_image(screen, depth_affected_image)
        clock.tick(60)  # Increased FPS limit to 60

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
