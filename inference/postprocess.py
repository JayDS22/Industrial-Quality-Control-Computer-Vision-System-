
#!/usr/bin/env python3
"""
Post-processing Module for Quality Control Results
Refines and validates detection and segmentation results
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
from sklearn.cluster import DBSCAN
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class PostProcessor:
    """Post-processing for quality control results"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize post-processor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.quality_thresholds = self.config.get('quality_control', {}).get('thresholds', {})
        
    def process_results(self, 
                       ensemble_results: Dict,
                       segmentation_results: Dict,
                       image_shape: Tuple) -> Dict:
        """
        Post-process ensemble and segmentation results
        
        Args:
            ensemble_results: Results from ensemble predictor
            segmentation_results: Results from segmentation
            image_shape: Shape of input image
            
        Returns:
            Processed and refined results
        """
        try:
            processed_results = {
                'detections': [],
                'quality_assessment': {},
                'risk_analysis': {},
                'recommendations': [],
                'metadata': {
                    'processing_timestamp': datetime.now().isoformat(),
                    'image_shape': image_shape,
                    'post_processing_version': '1.0.0'
                }
            }
            
            # Get raw detections
            raw_detections = ensemble_results.get('detections', [])
            
            if not raw_detections:
                processed_results['quality_assessment'] = self._assess_no_defects()
                return processed_results
            
            # Refine detections
            refined_detections = self._refine_detections(
                raw_detections, 
                segmentation_results,
                image_shape
            )
            
            # Filter detections
            filtered_detections = self._filter_detections(refined_detections)
            
            # Merge overlapping detections
            merged_detections = self._merge_overlapping_detections(filtered_detections)
            
            # Validate detections
            validated_detections = self._validate_detections(merged_detections, image_shape)
            
            # Assess overall quality
            quality_assessment = self._assess_quality(validated_detections, segmentation_results)
            
            # Risk analysis
            risk_analysis = self._analyze_risks(validated_detections, quality_assessment)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                validated_detections, 
                quality_assessment, 
                risk_analysis
            )
            
            processed_results.update({
                'detections': validated_detections,
                'quality_assessment': quality_assessment,
                'risk_analysis': risk_analysis,
                'recommendations': recommendations
            })
            
            logger.info(f"Post-processed {len(validated_detections)} detections")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in post-processing: {str(e)}")
            return {'error': str(e)}
    
    def _refine_detections(self, 
                          detections: List[Dict],
                          segmentation_results: Dict,
                          image_shape: Tuple) -> List[Dict]:
        """
        Refine detections using segmentation information
        
        Args:
            detections: Raw detections
            segmentation_results: Segmentation results
            image_shape: Image shape
            
        Returns:
            Refined detections
        """
        refined = []
        segmented_regions = segmentation_results.get('segmented_regions', [])
        
        for i, detection in enumerate(detections):
            refined_detection = detection.copy()
            
            # Find corresponding segmentation region
            seg_region = None
            for region in segmented_regions:
                if region.get('detection_id') == i:
                    seg_region = region
                    break
            
            if seg_region:
                # Update with segmentation information
                refined_detection.update({
                    'segmentation_confidence': seg_region.get('confidence_score', 0.0),
                    'area_pixels': seg_region.get('area_pixels', 0),
                    'area_percentage': seg_region.get('area_percentage', 0.0),
                    'perimeter': seg_region.get('perimeter', 0.0),
                    'compactness': seg_region.get('compactness', 0.0),
                    'contour_count': len(seg_region.get('contours', [])),
                    'has_segmentation': True
                })
                
                # Refine bounding box based on actual segmentation
                refined_bbox = self._refine_bbox_from_mask(
                    detection['bbox'],
                    seg_region.get('mask'),
                    image_shape
                )
                refined_detection['bbox'] = refined_bbox
                
                # Update severity based on segmentation
                refined_severity = self._refine_severity(
                    detection,
                    seg_region
                )
                refined_detection['final_severity'] = refined_severity
                
            else:
                # No segmentation available
                refined_detection.update({
                    'segmentation_confidence': 0.0,
                    'area_pixels': self._estimate_area_from_bbox(detection['bbox']),
                    'area_percentage': 0.0,
                    'perimeter': 0.0,
                    'compactness': 0.0,
                    'contour_count': 0,
                    'has_segmentation': False
                })
                refined_detection['final_severity'] = detection.get('severity', 'minor')
            
            # Add geometric features
            refined_detection.update(self._calculate_geometric_features(refined_detection))
            
            refined.append(refined_detection)
        
        return refined
    
    def _filter_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Filter detections based on confidence and quality criteria
        
        Args:
            detections: Input detections
            
        Returns:
            Filtered detections
        """
        filtered = []
        
        for detection in detections:
            # Confidence filtering
            confidence = detection.get('ensemble_confidence', detection.get('confidence', 0.0))
            min_confidence = self.quality_thresholds.get('confidence_threshold', 0.5)
            
            if confidence < min_confidence:
                logger.debug(f"Filtered detection with low confidence: {confidence:.3f}")
                continue
            
            # Area filtering
            area_percentage = detection.get('area_percentage', 0.0)
            max_area_threshold = self.quality_thresholds.get('area_threshold_percent', 50.0)
            
            if area_percentage > max_area_threshold:
                logger.debug(f"Filtered detection with large area: {area_percentage:.1f}%")
                continue
            
            # Aspect ratio filtering (remove extremely elongated detections)
            bbox = detection['bbox']
            aspect_ratio = bbox['width'] / bbox['height'] if bbox['height'] > 0 else float('inf')
            
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                logger.debug(f"Filtered detection with extreme aspect ratio: {aspect_ratio:.2f}")
                continue
            
            # Size filtering (remove very small detections)
            if bbox['width'] < 5 or bbox['height'] < 5:
                logger.debug("Filtered very small detection")
                continue
            
            filtered.append(detection)
        
        logger.info(f"Filtered {len(detections) - len(filtered)} detections")
        return filtered
    
    def _merge_overlapping_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Merge overlapping detections of the same class
        
        Args:
            detections: Input detections
            
        Returns:
            Merged detections
        """
        if len(detections) <= 1:
            return detections
        
        # Group by class
        class_groups = {}
        for detection in detections:
            class_name = detection['class']
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(detection)
        
        merged = []
        
        for class_name, class_detections in class_groups.items():
            if len(class_detections) == 1:
                merged.extend(class_detections)
                continue
            
            # Calculate IoU matrix
            n = len(class_detections)
            iou_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i + 1, n):
                    iou = self._calculate_iou(
                        class_detections[i]['bbox'],
                        class_detections[j]['bbox']
                    )
                    iou_matrix[i, j] = iou
                    iou_matrix[j, i] = iou
            
            # Find overlapping groups using clustering
            overlap_threshold = 0.3
            distance_matrix = 1.0 - iou_matrix
            
            # Use DBSCAN for clustering
            clustering = DBSCAN(
                metric='precomputed',
                eps=1.0 - overlap_threshold,
                min_samples=1
            )
            
            labels = clustering.fit_predict(distance_matrix)
            
            # Merge detections in each cluster
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                cluster_indices = np.where(labels == label)[0]
                
                if len(cluster_indices) == 1:
                    # Single detection
                    merged.append(class_detections[cluster_indices[0]])
                else:
                    # Merge multiple detections
                    cluster_detections = [class_detections[i] for i in cluster_indices]
                    merged_detection = self._merge_detection_cluster(cluster_detections)
                    merged.append(merged_detection)
        
        logger.info(f"Merged {len(detections)} detections into {len(merged)}")
        return merged
    
    def _merge_detection_cluster(self, detections: List[Dict]) -> Dict:
        """
        Merge a cluster of overlapping detections
        
        Args:
            detections: List of detections to merge
            
        Returns:
            Merged detection
        """
        # Use detection with highest confidence as base
        base_detection = max(detections, key=lambda d: d.get('ensemble_confidence', d.get('confidence', 0)))
        merged = base_detection.copy()
        
        # Calculate merged bounding box
        x1_min = min(d['bbox']['x1'] for d in detections)
        y1_min = min(d['bbox']['y1'] for d in detections)
        x2_max = max(d['bbox']['x2'] for d in detections)
        y2_max = max(d['bbox']['y2'] for d in detections)
        
        merged_bbox = {
            'x1': x1_min,
            'y1': y1_min,
            'x2': x2_max,
            'y2': y2_max,
            'width': x2_max - x1_min,
            'height': y2_max - y1_min,
            'center_x': (x1_min + x2_max) / 2,
            'center_y': (y1_min + y2_max) / 2
        }
        
        merged['bbox'] = merged_bbox
        
        # Average confidence scores
        confidences = [d.get('ensemble_confidence', d.get('confidence', 0)) for d in detections]
        merged['ensemble_confidence'] = np.mean(confidences)
        merged['confidence'] = np.mean(confidences)
        
        # Use maximum severity
        severities = [d.get('final_severity', d.get('severity', 'minor')) for d in detections]
        severity_order = {'minor': 1, 'major': 2, 'critical': 3}
        max_severity = max(severities, key=lambda s: severity_order.get(s, 0))
        merged['final_severity'] = max_severity
        
        # Sum areas if available
        areas = [d.get('area_pixels', 0) for d in detections if d.get('area_pixels', 0) > 0]
        if areas:
            merged['area_pixels'] = sum(areas)
        
        # Add merge information
        merged['merged_from'] = len(detections)
        merged['original_confidences'] = confidences
        
        return merged
    
    def _validate_detections(self, detections: List[Dict], image_shape: Tuple) -> List[Dict]:
        """
        Validate detections for consistency and plausibility
        
        Args:
            detections: Input detections
            image_shape: Image shape
            
        Returns:
            Validated detections
        """
        validated = []
        height, width = image_shape[:2]
        
        for detection in detections:
            # Validate bounding box
            bbox = detection['bbox']
            
            # Ensure bbox is within image bounds
            bbox['x1'] = max(0, min(bbox['x1'], width - 1))
            bbox['y1'] = max(0, min(bbox['y1'], height - 1))
            bbox['x2'] = max(bbox['x1'] + 1, min(bbox['x2'], width))
            bbox['y2'] = max(bbox['y1'] + 1, min(bbox['y2'], height))
            
            # Recalculate bbox properties
            bbox['width'] = bbox['x2'] - bbox['x1']
            bbox['height'] = bbox['y2'] - bbox['y1']
            bbox['center_x'] = (bbox['x1'] + bbox['x2']) / 2
            bbox['center_y'] = (bbox['y1'] + bbox['y2']) / 2
            
            # Validate area consistency
            bbox_area = bbox['width'] * bbox['height']
            reported_area = detection.get('area_pixels', bbox_area)
            
            # If segmentation area is much larger than bbox, there might be an error
            if reported_area > bbox_area * 2:
                logger.warning(f"Inconsistent area detected: segmented={reported_area}, bbox={bbox_area}")
                detection['area_pixels'] = bbox_area
                detection['area_consistency_warning'] = True
            
            # Add validation score
            validation_score = self._calculate_validation_score(detection, image_shape)
            detection['validation_score'] = validation_score
            
            # Only keep detections with reasonable validation scores
            if validation_score > 0.3:
                validated.append(detection)
            else:
                logger.debug(f"Filtered detection with low validation score: {validation_score:.3f}")
        
        return validated
    
    def _calculate_validation_score(self, detection: Dict, image_shape: Tuple) -> float:
        """
        Calculate validation score for detection
        
        Args:
            detection: Detection to validate
            image_shape: Image shape
            
        Returns:
            Validation score (0-1)
        """
        score = 1.0
        
        # Confidence component (30%)
        confidence = detection.get('ensemble_confidence', detection.get('confidence', 0))
        confidence_score = min(confidence / 0.8, 1.0)  # Normalize to 0.8 threshold
        
        # Size reasonableness (20%)
        bbox = detection['bbox']
        area_ratio = (bbox['width'] * bbox['height']) / (image_shape[0] * image_shape[1])
        size_score = 1.0 if 0.001 <= area_ratio <= 0.5 else 0.5
        
        # Aspect ratio reasonableness (20%)
        aspect_ratio = bbox['width'] / max(bbox['height'], 1)
        aspect_score = 1.0 if 0.2 <= aspect_ratio <= 5.0 else 0.5
        
        # Segmentation consistency (20%)
        if detection.get('has_segmentation', False):
            seg_confidence = detection.get('segmentation_confidence', 0)
            seg_score = min(seg_confidence / 0.5, 1.0)
        else:
            seg_score = 0.7  # Neutral score for no segmentation
        
        # Location reasonableness (10%)
        center_x = bbox['center_x'] / image_shape[1]
        center_y = bbox['center_y'] / image_shape[0]
        # Prefer defects not too close to edges
        location_score = min(
            min(center_x, 1 - center_x) / 0.05,
            min(center_y, 1 - center_y) / 0.05,
            1.0
        )
        
        # Weighted average
        total_score = (
            0.3 * confidence_score +
            0.2 * size_score +
            0.2 * aspect_score +
            0.2 * seg_score +
            0.1 * location_score
        )
        
        return total_score
    
    def _assess_quality(self, detections: List[Dict], segmentation_results: Dict) -> Dict:
        """
        Assess overall product quality based on detections
        
        Args:
            detections: Final detections
            segmentation_results: Segmentation results
            
        Returns:
            Quality assessment
        """
        if not detections:
            return self._assess_no_defects()
        
        # Count defects by severity
        severity_counts = {'critical': 0, 'major': 0, 'minor': 0}
        for detection in detections:
            severity = detection.get('final_severity', 'minor')
            severity_counts[severity] += 1
        
        # Apply quality rules
        quality_grade, pass_fail, risk_level = self._apply_quality_rules(severity_counts)
        
        # Calculate quality metrics
        total_defect_area = sum(d.get('area_pixels', 0) for d in detections)
        avg_confidence = np.mean([
            d.get('ensemble_confidence', d.get('confidence', 0)) 
            for d in detections
        ])
        
        # Quality assessment
        assessment = {
            'quality_grade': quality_grade,
            'pass_fail_status': pass_fail,
            'risk_level': risk_level,
            'total_defects': len(detections),
            'severity_breakdown': severity_counts,
            'total_defect_area_pixels': total_defect_area,
            'defect_density': segmentation_results.get('defect_density', 0.0),
            'average_confidence': avg_confidence,
            'quality_score': self._calculate_quality_score(severity_counts, avg_confidence),
            'meets_requirements': pass_fail == 'PASS',
            'recommended_action': self._get_recommended_action(pass_fail, risk_level)
        }
        
        return assessment
    
    def _assess_no_defects(self) -> Dict:
        """Return assessment for no defects detected"""
        return {
            'quality_grade': 'A',
            'pass_fail_status': 'PASS',
            'risk_level': 'low',
            'total_defects': 0,
            'severity_breakdown': {'critical': 0, 'major': 0, 'minor': 0},
            'total_defect_area_pixels': 0,
            'defect_density': 0.0,
            'average_confidence': 1.0,
            'quality_score': 100.0,
            'meets_requirements': True,
            'recommended_action': 'accept'
        }
    
    def _apply_quality_rules(self, severity_counts: Dict) -> Tuple[str, str, str]:
        """
        Apply quality control rules
        
        Args:
            severity_counts: Count of defects by severity
            
        Returns:
            Tuple of (grade, pass_fail, risk_level)
        """
        critical = severity_counts['critical']
        major = severity_counts['major']
        minor = severity_counts['minor']
        
        # Get thresholds from config
        critical_limit = self.quality_thresholds.get('critical_defect_limit', 0)
        major_limit = self.quality_thresholds.get('major_defect_limit', 1)
        minor_limit = self.quality_thresholds.get('minor_defect_limit', 3)
        
        # Apply rules
        if critical > critical_limit:
            return 'F', 'FAIL', 'high'
        elif major > major_limit:
            return 'D', 'FAIL', 'high'
        elif minor > minor_limit:
            return 'C', 'CONDITIONAL', 'medium'
        elif major > 0:
            return 'B', 'CONDITIONAL', 'low'
        else:
            return 'A', 'PASS', 'low'
    
    def _calculate_quality_score(self, severity_counts: Dict, avg_confidence: float) -> float:
        """
        Calculate numerical quality score (0-100)
        
        Args:
            severity_counts: Defect counts by severity
            avg_confidence: Average detection confidence
            
        Returns:
            Quality score
        """
        base_score = 100.0
        
        # Penalty for defects
        penalties = {
            'critical': 30,
            'major': 15,
            'minor': 5
        }
        
        for severity, count in severity_counts.items():
            base_score -= count * penalties[severity]
        
        # Adjust for confidence
        confidence_factor = min(avg_confidence / 0.8, 1.0)
        
        final_score = max(0, base_score * confidence_factor)
        return final_score
    
    def _get_recommended_action(self, pass_fail: str, risk_level: str) -> str:
        """Get recommended action based on quality assessment"""
        action_mapping = {
            ('PASS', 'low'): 'accept',
            ('CONDITIONAL', 'low'): 'accept_with_monitoring', 
            ('CONDITIONAL', 'medium'): 'review_required',
            ('FAIL', 'high'): 'reject',
            ('FAIL', 'medium'): 'rework_required'
        }
        
        return action_mapping.get((pass_fail, risk_level), 'manual_inspection')
    
    def _analyze_risks(self, detections: List[Dict], quality_assessment: Dict) -> Dict:
        """
        Analyze risks associated with detected defects
        
        Args:
            detections: Final detections
            quality_assessment: Quality assessment
            
        Returns:
            Risk analysis
        """
        risk_factors = []
        risk_score = 0.0
        
        # Analyze defect patterns
        if len(detections) > 0:
            # Clustering analysis
            cluster_risk = self._analyze_defect_clustering(detections)
            if cluster_risk['has_clusters']:
                risk_factors.append("Clustered defects detected - possible systematic issue")
                risk_score += 0.3
            
            # Size analysis
            large_defects = [d for d in detections if d.get('area_pixels', 0) > 1000]
            if large_defects:
                risk_factors.append(f"{len(large_defects)} large defects detected")
                risk_score += 0.2 * len(large_defects)
            
            # Critical defect analysis
            critical_defects = [d for d in detections if d.get('final_severity') == 'critical']
            if critical_defects:
                risk_factors.append("Critical defects present - immediate attention required")
                risk_score += 0.5 * len(critical_defects)
        
        # Overall risk level
        if risk_score >= 1.0:
            overall_risk = 'high'
        elif risk_score >= 0.5:
            overall_risk = 'medium'
        else:
            overall_risk = 'low'
        
        return {
            'overall_risk_level': overall_risk,
            'risk_score': min(risk_score, 1.0),
            'risk_factors': risk_factors,
            'defect_clustering': cluster_risk,
            'requires_immediate_action': quality_assessment['pass_fail_status'] == 'FAIL'
        }
    
    def _analyze_defect_clustering(self, detections: List[Dict]) -> Dict:
        """
        Analyze if defects are clustered (indicating systematic issues)
        
        Args:
            detections: List of detections
            
        Returns:
            Clustering analysis
        """
        if len(detections) < 3:
            return {'has_clusters': False, 'cluster_count': 0}
        
        # Extract center points
        points = np.array([
            [d['bbox']['center_x'], d['bbox']['center_y']] 
            for d in detections
        ])
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=100, min_samples=2)  # 100 pixel radius
        labels = clustering.fit_predict(points)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        has_clusters = n_clusters > 0
        
        return {
            'has_clusters': has_clusters,
            'cluster_count': n_clusters,
            'clustered_defects': np.sum(labels >= 0),
            'isolated_defects': np.sum(labels == -1)
        }
    
    def _generate_recommendations(self, 
                                detections: List[Dict],
                                quality_assessment: Dict,
                                risk_analysis: Dict) -> List[str]:
        """
        Generate actionable recommendations
        
        Args:
            detections: Final detections
            quality_assessment: Quality assessment
            risk_analysis: Risk analysis
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Quality-based recommendations
        if quality_assessment['pass_fail_status'] == 'FAIL':
            recommendations.append("REJECT: Product does not meet quality standards")
            
            if quality_assessment['severity_breakdown']['critical'] > 0:
                recommendations.append("Critical defects detected - investigate root cause immediately")
            
            if quality_assessment['severity_breakdown']['major'] > 2:
                recommendations.append("Multiple major defects - review manufacturing process")
        
        elif quality_assessment['pass_fail_status'] == 'CONDITIONAL':
            recommendations.append("CONDITIONAL PASS: Monitor closely and consider rework")
            recommendations.append("Increase inspection frequency for similar products")
        
        # Risk-based recommendations
        if risk_analysis['overall_risk_level'] == 'high':
            recommendations.append("High risk detected - implement immediate corrective actions")
        
        if risk_analysis['defect_clustering']['has_clusters']:
            recommendations.append("Defect clustering detected - check for systematic manufacturing issues")
        
        # Defect-specific recommendations
        defect_types = set(d['class'] for d in detections)
        for defect_type in defect_types:
            type_recs = self._get_defect_type_recommendations(defect_type)
            recommendations.extend(type_recs)
        
        # Process improvement recommendations
        if quality_assessment['average_confidence'] < 0.7:
            recommendations.append("Low detection confidence - consider additional inspection methods")
        
        if quality_assessment['defect_density'] > 5.0:
            recommendations.append("High defect density - review entire manufacturing process")
        
        return recommendations
    
    def _get_defect_type_recommendations(self, defect_type: str) -> List[str]:
        """Get specific recommendations for defect types"""
        recommendations_map = {
            'crack': [
                "Check material stress levels and handling procedures",
                "Verify temperature control during manufacturing"
            ],
            'scratch': [
                "Review handling and packaging procedures",
                "Check for abrasive contact points in production line"
            ],
            'dent': [
                "Inspect handling equipment for damage",
                "Review impact protection during transport"
            ],
            'discoloration': [
                "Check chemical process parameters",
                "Verify environmental conditions (humidity, temperature)"
            ],
            'contamination': [
                "Review cleaning procedures and protocols",
                "Check for foreign material sources in production area"
            ]
        }
        
        return recommendations_map.get(defect_type, [])
    
    def _refine_bbox_from_mask(self, 
                             original_bbox: Dict, 
                             mask: np.ndarray,
                             image_shape: Tuple) -> Dict:
        """
        Refine bounding box based on segmentation mask
        
        Args:
            original_bbox: Original bounding box
            mask: Segmentation mask
            image_shape: Image shape
            
        Returns:
            Refined bounding box
        """
        if mask is None:
            return original_bbox
        
        try:
            # Find actual defect bounds in mask
            coords = np.where(mask > 0)
            
            if len(coords[0]) == 0:
                return original_bbox
            
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Add small padding
            padding = 5
            y_min = max(0, y_min - padding)
            x_min = max(0, x_min - padding)
            y_max = min(image_shape[0], y_max + padding)
            x_max = min(image_shape[1], x_max + padding)
            
            return {
                'x1': x_min,
                'y1': y_min,
                'x2': x_max,
                'y2': y_max,
                'width': x_max - x_min,
                'height': y_max - y_min,
                'center_x': (x_min + x_max) / 2,
                'center_y': (y_min + y_max) / 2
            }
            
        except Exception as e:
            logger.error(f"Error refining bbox from mask: {str(e)}")
            return original_bbox
    
    def _refine_severity(self, detection: Dict, seg_region: Dict) -> str:
        """
        Refine severity assessment using segmentation information
        
        Args:
            detection: Original detection
            seg_region: Segmentation region
            
        Returns:
            Refined severity level
        """
        original_severity = detection.get('severity', 'minor')
        area_percentage = seg_region.get('area_percentage', 0.0)
        compactness = seg_region.get('compactness', 0.0)
        
        # Rules for severity escalation
        if area_percentage > 5.0:  # Large defect
            if original_severity == 'minor':
                return 'major'
            elif original_severity == 'major':
                return 'critical'
        
        if compactness < 0.3:  # Irregular shape (possible crack propagation)
            if detection['class'] in ['crack', 'scratch'] and original_severity == 'minor':
                return 'major'
        
        return original_severity
    
    def _estimate_area_from_bbox(self, bbox: Dict) -> int:
        """Estimate area from bounding box (fallback when no segmentation)"""
        return int(bbox['width'] * bbox['height'] * 0.6)  # Assume 60% fill ratio
    
    def _calculate_geometric_features(self, detection: Dict) -> Dict:
        """Calculate additional geometric features"""
        bbox = detection['bbox']
        
        return {
            'aspect_ratio': bbox['width'] / max(bbox['height'], 1),
            'bbox_area': bbox['width'] * bbox['height'],
            'normalized_position': {
                'x': bbox['center_x'] / 1000,  # Normalize assuming max 1000px width
                'y': bbox['center_y'] / 1000   # Normalize assuming max 1000px height
            }
        }
    
    def _calculate_iou(self, bbox1: Dict, bbox2: Dict) -> float:
        """Calculate Intersection over Union of two bounding boxes"""
        # Calculate intersection
        x1 = max(bbox1['x1'], bbox2['x1'])
        y1 = max(bbox1['y1'], bbox2['y1'])
        x2 = min(bbox1['x2'], bbox2['x2'])
        y2 = min(bbox1['y2'], bbox2['y2'])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = bbox1['width'] * bbox1['height']
        area2 = bbox2['width'] * bbox2['height']
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update_config(self, new_config: Dict):
        """Update configuration"""
        self.config.update(new_config)
        self.quality_thresholds = self.config.get('quality_control', {}).get('thresholds', {})
