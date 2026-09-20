import math
from typing import List, Tuple, Dict, Any, Optional
from shapely.geometry import Point as ShapelyPoint, Polygon as ShapelyPolygon, LineString
from uav_neurosym.schema import PolygonGeofence, NoFlyZone, Point3D


def check_geometry_feasibility(
    proposed_path: List[Point3D],
    airspace: PolygonGeofence,
    no_fly_zones: List[NoFlyZone],
    min_separation_m: float = 10.0
) -> Tuple[bool, str, float, Dict[str, Any]]:
    """
    Evaluates spatial consistency, geofence bounds, and NFZ avoidance (Eq. 15, 17, 19).
    Returns (is_valid, error_category, error_magnitude, details_dict).
    """
    metrics: Dict[str, Any] = {}

    if not proposed_path or len(proposed_path) == 0:
        return False, "EMPTY_PATH", 0.0, {"detail": "Proposed path has no waypoints"}

    # Construct Airspace Geofence Polygon
    airspace_poly = ShapelyPolygon(airspace.vertices)

    # 1. Check Altitude and Geofence Containment for each waypoint
    for idx, wp in enumerate(proposed_path):
        pt_2d = ShapelyPoint(wp.x, wp.y)
        
        # Check lateral geofence
        if not airspace_poly.contains(pt_2d) and not airspace_poly.touches(pt_2d):
            dist_outside = pt_2d.distance(airspace_poly)
            return False, "GEOFENCE_LATERAL_BREACH", dist_outside, {
                "waypoint_index": idx,
                "waypoint": wp.dict(),
                "distance_outside_m": round(dist_outside, 2)
            }

        # Check altitude boundaries
        if wp.z < airspace.min_alt_m:
            err = airspace.min_alt_m - wp.z
            return False, "ALTITUDE_BELOW_MINIMUM", err, {
                "waypoint_index": idx,
                "waypoint_alt": wp.z,
                "min_alt": airspace.min_alt_m
            }

        if wp.z > airspace.max_alt_m:
            err = wp.z - airspace.max_alt_m
            return False, "ALTITUDE_EXCEEDED_CEILING", err, {
                "waypoint_index": idx,
                "waypoint_alt": wp.z,
                "max_alt": airspace.max_alt_m
            }

    # 2. Check Line Segment Intersections with Airspace & NFZs
    if len(proposed_path) >= 2:
        path_line = LineString([(p.x, p.y) for p in proposed_path])

        # Check path string vs NFZs
        for nfz in no_fly_zones:
            if not nfz.is_active:
                continue

            nfz_poly = ShapelyPolygon(nfz.polygon)
            
            # Check 2D intersection
            if path_line.intersects(nfz_poly):
                # Verify 3D altitude overlap
                min_path_z = min(p.z for p in proposed_path)
                max_path_z = max(p.z for p in proposed_path)

                if max_path_z >= nfz.min_alt_m and min_path_z <= nfz.max_alt_m:
                    penetration_depth = path_line.intersection(nfz_poly).length
                    return False, "NO_FLY_ZONE_PENETRATION", penetration_depth, {
                        "nfz_id": nfz.nfz_id,
                        "nfz_name": nfz.name,
                        "penetration_length_m": round(penetration_depth, 2)
                    }

    metrics["waypoints_checked"] = len(proposed_path)
    metrics["geofence_name"] = airspace.name
    metrics["active_nfzs"] = len([n for n in no_fly_zones if n.is_active])

    return True, "", 0.0, metrics
