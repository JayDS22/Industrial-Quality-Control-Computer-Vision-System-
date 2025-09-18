
#!/usr/bin/env python3
"""
Image Segmentation Module for Quality Control
Provides detailed defect area analysis and masking
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from skimage import measure, morphology
from skimage.filters import threshold_otsu, gaussian
from skimage.segmentation import watershed
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

logger = logging.getLogger(__name__)

class ImageSegmentator:
    """Advanced image segmentation for defect analysis"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize image segmentator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.segmentation_methods = {
            'threshold': self._threshold_segmentation,
            'watershed': self._watershed_segmentation,
            'adaptive': self._adaptive_segmentation,
            'region_growing': self._region_growing_segmentation
        }
    
    def segment_defects(self, 
                       image: np.ndarray, 
                       detections: List[Dict]) -> Dict:
        """
        Segment defects in detected regions
        
        Args:
            image: Input image
            detections: List of detection results
            
        Returns:
            Segmentation results
        """
        try:
            segmentation_results = {
                'segmented_regions': [],
                'masks': [],
                'contours': [],
                'area_analysis': {},
                'total_defect_area': 0,
                'defect_density': 0.0
            }
            
            if not detections:
                return segmentation_results
            
            # Convert image to grayscale if needed
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = image.copy()
            
            total_image_area = gray_image.shape[0] * gray_image.shape[1]
            total_defect_area = 0
            
            for i, detection in enumerate(detections):
                # Extract detection region
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                
                # Extract region of interest
                roi = gray_image[y1:y2, x1:x2]
                
                if roi.size == 0:
                    continue
                
                # Segment the defect region
                segmentation_result = self._segment_region(
                    roi, 
                    detection['class'],
                    detection.get('confidence', 0.5)
                )
                
                # Convert local coordinates to global
                global_mask = np.zeros_like(gray_image, dtype=np.uint8)
                if segmentation_result['mask'] is not None:
                    global_mask[y1:y2, x1:x2] = segmentation_result['mask']
                
                # Calculate area metrics
                defect_area = np.sum(segmentation_result['mask'] > 0) if segmentation_result['mask'] is not None else 0
                total_defect_area += defect_area
                
                # Find contours
                contours = self._find_contours(segmentation_result['mask'])
                
                # Adjust contour coordinates to global image
                global_contours = []
                for contour in contours:
                    global_contour = contour.copy()
                    global_contour[:, 0, 0] += x1  # Adjust x coordinates
                    global_contour[:, 0, 1] += y1  # Adjust y coordinates
                    global_contours.append(global_contour)
                
                # Store results
                region_result = {
                    'detection_id': i,
                    'defect_class': detection['class'],
                    'confidence': detection.get('confidence', 0.0),
                    'bbox': bbox,
                    'mask': global_mask,
                    'local_mask': segmentation_result['mask'],
                    'contours': global_contours,
                    'area_pixels': defect_area,
                    'area_percentage': (defect_area / total_image_area) * 100,
                    'perimeter': self._calculate_perimeter(contours),
                    'compactness': self._calculate_compactness(defect_area, contours),
                    'segmentation_method': segmentation_result['method'],
                    'confidence_score': segmentation_result['confidence']
                }
                
                segmentation_results['segmented_regions'].append(region_result)
                segmentation_results['masks'].append(global_mask)
                segmentation_results['contours'].extend(global_contours)
            
            # Overall statistics
            segmentation_results['total_defect_area'] = total_defect_area
            segmentation_results['defect_density'] = (total_defect_area / total_image_area) * 100
            segmentation_results['area_analysis'] = self._analyze_defect_areas(
                segmentation_results['segmented_regions']
            )
            
            logger.info(f"Segmented {len(detections)} regions, total area: {total_defect_area} pixels")
            
            return segmentation_results
            
        except Exception as e:
            logger.error(f"Error in defect segmentation: {str(e)}")
            return {'error': str(e)}
    
    def _segment_region(self, 
                       roi: np.ndarray, 
                       defect_class: str, 
                       confidence: float) -> Dict:
        """
        Segment individual defect region
        
        Args:
            roi: Region of interest
            defect_class: Type of defect
            confidence: Detection confidence
            
        Returns:
            Segmentation result
        """
        # Choose segmentation method based on defect type
        method = self._select_segmentation_method(defect_class)
        
        try:
            mask, seg_confidence = self.segmentation_methods[method](roi, defect_class)
            
            return {
                'mask': mask,
                'method': method,
                'confidence': seg_confidence
            }
            
        except Exception as e:
            logger.error(f"Error segmenting region with method {method}: {str(e)}")
            # Fallback to simple thresholding
            mask, seg_confidence = self._threshold_segmentation(roi, defect_class)
            return {
                'mask': mask,
                'method': 'threshold_fallback',
                'confidence': seg_confidence
            }
    
    def _select_segmentation_method(self, defect_class: str) -> str:
        """Select appropriate segmentation method for defect type"""
        method_mapping = {
            'crack': 'adaptive',
            'scratch': 'adaptive', 
            'dent': 'watershed',
            'discoloration': 'threshold',
            'contamination': 'region_growing'
        }
        
        return method_mapping.get(defect_class, 'threshold')
    
    def _threshold_segmentation(self, 
                              roi: np.ndarray, 
                              defect_class: str) -> Tuple[np.ndarray, float]:
        """
        Threshold-based segmentation
        
        Args:
            roi: Region of interest
            defect_class: Type of defect
            
        Returns:
            Binary mask and confidence score
        """
        try:
            # Apply Gaussian blur to reduce noise
            blurred = gaussian(roi, sigma=1.0)
            
            # Calculate threshold using Otsu's method
            threshold = threshold_otsu(blurred)
            
            # Adjust threshold based on defect type
            threshold_adjustments = {
                'crack': 0.9,
                'scratch': 0.9,
                'dent': 1.1,
                'discoloration': 0.8,
                'contamination': 0.85
            }
            
            adjusted_threshold = threshold * threshold_adjustments.get(defect_class, 1.0)
            
            # Create binary mask
            if defect_class in ['crack', 'scratch']:
                # Dark defects
                mask = blurred < adjusted_threshold
            else:
                # Light defects or discoloration
                mask = blurred > adjusted_threshold
            
            # Clean up mask
            mask = self._clean_mask(mask)
            
            # Calculate confidence based on separation
            confidence = self._calculate_segmentation_confidence(roi, mask, threshold)
            
            return mask.astype(np.uint8) * 255, confidence
            
        except Exception as e:
            logger.error(f"Error in threshold segmentation: {str(e)}")
            return np.zeros_like(roi, dtype=np.uint8), 0.0
    
    def _adaptive_segmentation(self, 
                             roi: np.ndarray, 
                             defect_class: str) -> Tuple[np.ndarray, float]:
        """
        Adaptive threshold segmentation for linear defects
        
        Args:
            roi: Region of interest
            defect_class: Type of defect
            
        Returns:
            Binary mask and confidence score
        """
        try:
            # Apply adaptive thresholding
            block_size = max(11, min(roi.shape) // 10)
            if block_size % 2 == 0:
                block_size += 1
            
            # Convert to uint8 if needed
            if roi.dtype != np.uint8:
                roi_uint8 = ((roi - roi.min()) / (roi.max() - roi.min()) * 255).astype(np.uint8)
            else:
                roi_uint8 = roi
            
            # Apply adaptive threshold
            mask = cv2.adaptiveThreshold(
                roi_uint8,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV if defect_class in ['crack', 'scratch'] else cv2.THRESH_BINARY,
                block_size,
                2
            )
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Remove small components
            mask = self._remove_small_components(mask, min_size=20)
            
            # Calculate confidence
            confidence = self._calculate_adaptive_confidence(roi_uint8, mask)
            
            return mask, confidence
            
        except Exception as e:
            logger.error(f"Error in adaptive segmentation: {str(e)}")
            return np.zeros_like(roi, dtype=np.uint8), 0.0
    
    def _watershed_segmentation(self, 
                              roi: np.ndarray, 
                              defect_class: str) -> Tuple[np.ndarray, float]:
        """
        Watershed segmentation for blob-like defects
        
        Args:
            roi: Region of interest
            defect_class: Type of defect
            
        Returns:
            Binary mask and confidence score
        """
        try:
            # Preprocess image
            blurred = gaussian(roi, sigma=1.0)
            
            # Find local minima and maxima
            if defect_class == 'dent':
                # For dents, look for local minima
                local_minima = morphology.local_minima(blurred)
                markers = measure.label(local_minima)
            else:
                # For other defects, look for local maxima
                local_maxima = morphology.local_maxima(blurred)
                markers = measure.label(local_maxima)
            
            # Apply watershed
            if defect_class == 'dent':
                labels = watershed(blurred, markers)
            else:
                labels = watershed(-blurred, markers)
            
            # Create binary mask from watershed result
            mask = labels > 0
            
            # Clean up mask
            mask = self._clean_mask(mask)
            
            # Calculate confidence
            confidence = self._calculate_watershed_confidence(roi, mask, markers)
            
            return mask.astype(np.uint8) * 255, confidence
            
        except Exception as e:
            logger.error(f"Error in watershed segmentation: {str(e)}")
            return np.zeros_like(roi, dtype=np.uint8), 0.0
    
    def _region_growing_segmentation(self, 
                                   roi: np.ndarray, 
                                   defect_class: str) -> Tuple[np.ndarray, float]:
        """
        Region growing segmentation for contamination
        
        Args:
            roi: Region of interest
            defect_class: Type of defect
            
        Returns:
            Binary mask and confidence score
        """
        try:
            # Find seed points
            seeds = self._find_seed_points(roi, defect_class)
            
            if len(seeds) == 0:
                return np.zeros_like(roi, dtype=np.uint8), 0.0
            
            # Region growing
            mask = np.zeros_like(roi, dtype=bool)
            threshold = np.std(roi) * 2  # Adaptive threshold
            
            for seed in seeds:
                region_mask = self._grow_region(roi, seed, threshold)
                mask = mask | region_mask
            
            # Clean up mask
            mask = self._clean_mask(mask)
            
            # Calculate confidence
            confidence = self._calculate_region_growing_confidence(roi, mask, seeds)
            
            return mask.astype(np.uint8) * 255, confidence
            
        except Exception as e:
            logger.error(f"Error in region growing segmentation: {str(e)}")
            return np.zeros_like(roi, dtype=np.uint8), 0.0
    
    def _find_seed_points(self, roi: np.ndarray, defect_class: str) -> List[Tuple[int, int]]:
        """Find seed points for region growing"""
        # Apply Gaussian blur
        blurred = gaussian(roi, sigma=1.0)
        
        # Find extreme values based on defect type
        if defect_class == 'contamination':
            # Look for bright spots
            threshold = np.mean(blurred) + 2 * np.std(blurred)
            candidates = np.where(blurred > threshold)
        else:
            # Look for dark spots
            threshold = np.mean(blurred) - 2 * np.std(blurred)
            candidates = np.where(blurred < threshold)
        
        # Convert to list of tuples
        seeds = list(zip(candidates[0], candidates[1]))
        
        # Limit number of seeds
        if len(seeds) > 10:
            # Sample seeds uniformly
            step = len(seeds) // 10
            seeds = seeds[::step]
        
        return seeds
    
    def _grow_region(self, 
                    image: np.ndarray, 
                    seed: Tuple[int, int], 
                    threshold: float) -> np.ndarray:
        """Grow region from seed point"""
        h, w = image.shape
        mask = np.zeros_like(image, dtype=bool)
        
        # Initialize with seed
        y, x = seed
        if y < 0 or y >= h or x < 0 or x >= w:
            return mask
        
        seed_value = image[y, x]
        stack = [(y, x)]
        mask[y, x] = True
        
        # 8-connectivity neighbors
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        while stack:
            cy, cx = stack.pop()
            
            for dy, dx in neighbors:
                ny, nx = cy + dy, cx + dx
                
                if (0 <= ny < h and 0 <= nx < w and 
                    not mask[ny, nx] and 
                    abs(image[ny, nx] - seed_value) < threshold):
                    
                    mask[ny, nx] = True
                    stack.append((ny, nx))
        
        return mask
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean up binary mask"""
        # Remove small components
        mask = self._remove_small_components(mask, min_size=10)
        
        # Fill small holes
        mask = ndimage.binary_fill_holes(mask)
        
        # Smooth boundaries
        kernel = morphology.disk(2)
        mask = morphology.binary_closing(mask, kernel)
        mask = morphology.binary_opening(mask, kernel)
        
        return mask
    
    def _remove_small_components(self, mask: np.ndarray, min_size: int = 20) -> np.ndarray:
        """Remove small connected components"""
        if mask.dtype == bool:
            mask_uint8 = mask.astype(np.uint8) * 255
        else:
            mask_uint8 = mask
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(mask_uint8)
        
        # Calculate component sizes
        sizes = np.bincount(labels.ravel())
        
        # Create mask for large components
        large_components = sizes >= min_size
        large_components[0] = False  # Background
        
        # Filter mask
        new_mask = np.isin(labels, np.where(large_components)[0])
        
        return new_mask.astype(np.uint8) * 255 if mask.dtype != bool else new_mask
    
    def _find_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """Find contours in binary mask"""
        if mask is None:
            return []
        
        try:
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filter small contours
            min_contour_area = 10
            filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
            
            return filtered_contours
            
        except Exception as e:
            logger.error(f"Error finding contours: {str(e)}")
            return []
    
    def _calculate_perimeter(self, contours: List[np.ndarray]) -> float:
        """Calculate total perimeter of contours"""
        total_perimeter = 0.0
        for contour in contours:
            total_perimeter += cv2.arcLength(contour, True)
        return total_perimeter
    
    def _calculate_compactness(self, area: float, contours: List[np.ndarray]) -> float:
        """Calculate compactness (4π × Area / Perimeter²)"""
        if not contours or area == 0:
            return 0.0
        
        perimeter = self._calculate_perimeter(contours)
        if perimeter == 0:
            return 0.0
        
        compactness = (4 * np.pi * area) / (perimeter ** 2)
        return min(compactness, 1.0)  # Clamp to [0, 1]
    
    def _calculate_segmentation_confidence(self, 
                                         roi: np.ndarray, 
                                         mask: np.ndarray, 
                                         threshold: float) -> float:
        """Calculate segmentation confidence based on threshold separation"""
        try:
            mask_bool = mask > 0
            
            if not np.any(mask_bool):
                return 0.0
            
            foreground = roi[mask_bool]
            background = roi[~mask_bool]
            
            if len(foreground) == 0 or len(background) == 0:
                return 0.0
            
            # Calculate separation between foreground and background
            fg_mean = np.mean(foreground)
            bg_mean = np.mean(background)
            
            separation = abs(fg_mean - bg_mean) / (np.std(roi) + 1e-6)
            confidence = min(separation / 3.0, 1.0)  # Normalize
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating segmentation confidence: {str(e)}")
            return 0.0
    
    def _calculate_adaptive_confidence(self, 
                                     roi: np.ndarray, 
                                     mask: np.ndarray) -> float:
        """Calculate confidence for adaptive segmentation"""
        try:
            # Check mask quality
            mask_ratio = np.sum(mask > 0) / mask.size
            
            # Penalize very small or very large masks
            if mask_ratio < 0.01 or mask_ratio > 0.8:
                return 0.3
            
            # Check edge strength
            edges = cv2.Canny(roi, 50, 150)
            edge_overlap = np.sum((edges > 0) & (mask > 0)) / np.sum(mask > 0)
            
            confidence = 0.5 + 0.5 * edge_overlap
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating adaptive confidence: {str(e)}")
            return 0.5
    
    def _calculate_watershed_confidence(self, 
                                      roi: np.ndarray, 
                                      mask: np.ndarray, 
                                      markers: np.ndarray) -> float:
        """Calculate confidence for watershed segmentation"""
        try:
            num_regions = np.max(markers)
            mask_ratio = np.sum(mask > 0) / mask.size
            
            # Good segmentation should have reasonable number of regions and mask ratio
            region_score = 1.0 / (1.0 + abs(num_regions - 3))  # Prefer ~3 regions
            ratio_score = 1.0 - abs(mask_ratio - 0.2)  # Prefer ~20% coverage
            
            confidence = 0.5 * region_score + 0.5 * max(0, ratio_score)
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating watershed confidence: {str(e)}")
            return 0.5
    
    def _calculate_region_growing_confidence(self, 
                                           roi: np.ndarray, 
                                           mask: np.ndarray, 
                                           seeds: List[Tuple[int, int]]) -> float:
        """Calculate confidence for region growing segmentation"""
        try:
            if len(seeds) == 0:
                return 0.0
            
            # Check if grown regions make sense
            mask_ratio = np.sum(mask > 0) / mask.size
            
            # Confidence based on number of seeds and coverage
            seed_score = min(len(seeds) / 5.0, 1.0)  # Prefer 5 seeds
            coverage_score = min(mask_ratio * 5, 1.0)  # Prefer some coverage
            
            confidence = 0.6 * seed_score + 0.4 * coverage_score
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating region growing confidence: {str(e)}")
            return 0.5
    
    def _analyze_defect_areas(self, segmented_regions: List[Dict]) -> Dict:
        """Analyze defect areas and provide statistics"""
        if not segmented_regions:
            return {}
        
        areas = [region['area_pixels'] for region in segmented_regions]
        percentages = [region['area_percentage'] for region in segmented_regions]
        
        analysis = {
            'total_regions': len(segmented_regions),
            'total_area_pixels': sum(areas),
            'total_area_percentage': sum(percentages),
            'average_area_pixels': np.mean(areas),
            'median_area_pixels': np.median(areas),
            'max_area_pixels': max(areas),
            'min_area_pixels': min(areas),
            'area_std': np.std(areas),
            'size_distribution': self._categorize_defect_sizes(areas)
        }
        
        return analysis
    
    def _categorize_defect_sizes(self, areas: List[float]) -> Dict:
        """Categorize defects by size"""
        small = sum(1 for area in areas if area < 100)
        medium = sum(1 for area in areas if 100 <= area < 1000)
        large = sum(1 for area in areas if area >= 1000)
        
        return {
            'small_defects': small,
            'medium_defects': medium,
            'large_defects': large
        }
    
    def visualize_segmentation(self, 
                             image: np.ndarray, 
                             segmentation_results: Dict,
                             save_path: str = None) -> np.ndarray:
        """
        Visualize segmentation results
        
        Args:
            image: Original image
            segmentation_results: Segmentation results
            save_path: Path to save visualization
            
        Returns:
            Visualization image
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Original image
            axes[0, 0].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Combined mask
            combined_mask = np.zeros_like(image[:, :, 0] if len(image.shape) == 3 else image)
            for mask in segmentation_results.get('masks', []):
                combined_mask = np.maximum(combined_mask, mask)
            
            axes[0, 1].imshow(combined_mask, cmap='hot')
            axes[0, 1].set_title('Segmentation Masks')
            axes[0, 1].axis('off')
            
            # Overlay
            if len(image.shape) == 3:
                overlay = image.copy()
            else:
                overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Draw contours
            for region in segmentation_results.get('segmented_regions', []):
                contours = region.get('contours', [])
                color = self._get_defect_color(region['defect_class'])
                cv2.drawContours(overlay, contours, -1, color, 2)
            
            axes[1, 0].imshow(overlay)
            axes[1, 0].set_title('Segmentation Overlay')
            axes[1, 0].axis('off')
            
            # Statistics
            stats_text = self._format_statistics(segmentation_results)
            axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                           verticalalignment='top', fontsize=10, fontfamily='monospace')
            axes[1, 1].set_title('Segmentation Statistics')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Segmentation visualization saved to {save_path}")
            
            # Convert plot to image array
            fig.canvas.draw()
            vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            
            return vis_image
            
        except Exception as e:
            logger.error(f"Error creating segmentation visualization: {str(e)}")
            return image
    
    def _get_defect_color(self, defect_class: str) -> Tuple[int, int, int]:
        """Get color for defect class"""
        colors = {
            'crack': (255, 0, 0),      # Red
            'scratch': (0, 255, 0),    # Green
            'dent': (0, 0, 255),       # Blue
            'discoloration': (255, 255, 0),  # Yellow
            'contamination': (255, 0, 255)   # Magenta
        }
        return colors.get(defect_class, (128, 128, 128))  # Gray default
    
    def _format_statistics(self, segmentation_results: Dict) -> str:
        """Format statistics for display"""
        area_analysis = segmentation_results.get('area_analysis', {})
        
        stats = f"""Segmentation Results:
        
Total Regions: {area_analysis.get('total_regions', 0)}
Total Defect Area: {area_analysis.get('total_area_pixels', 0)} pixels
Coverage: {area_analysis.get('total_area_percentage', 0):.2f}%

Size Distribution:
- Small: {area_analysis.get('size_distribution', {}).get('small_defects', 0)}
- Medium: {area_analysis.get('size_distribution', {}).get('medium_defects', 0)}
- Large: {area_analysis.get('size_distribution', {}).get('large_defects', 0)}

Average Area: {area_analysis.get('average_area_pixels', 0):.1f} pixels
Max Area: {area_analysis.get('max_area_pixels', 0)} pixels
"""
        
        return stats
