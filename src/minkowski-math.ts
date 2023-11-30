// SPDX-License-Identifier: MIT
// Copyright (c) 2022, Symcode Oy
export interface Vec2 {
  x: number;
  y: number;
}

/**
 * Vec2Derived represents a Vec2 derived from 2 other Vec2s on polygons A and B.
 * Used for keeping references to original vertices where Minkowski sum polygon's
 * vertices where calculated from.
 */
export interface Vec2Derived extends Vec2 {
  // Index of vertex in polygon A
  refA: number;
  // Index of vertex in polygon B
  refB: number;
}

export interface Segment {
  a: Vec2;
  b: Vec2;
}

export interface ReusablePointArray {
  points: Vec2Derived[];
  pCount: number;
}

export interface ReusablePointArrayLp extends ReusablePointArray {
  iLowest: number;
}

export interface MinDistResult {
  // Distance
  d: number;
  // Point in polygon where the minimum distance was calculated to.
  minPoint: Vec2;
  // Indecies of "minimum distance" polygon vertices.
  // If minimum distance is between a point and a vertice v2i
  // is NaN; otherwise minimum distance is between a point and a
  // line segment and minPoint is between vertices in indices
  // v1i and v2i.
  v1i: number;
  v2i: number;
}

export function reusablePointArray2Array(
  ra: ReusablePointArray,
): Vec2Derived[] {
  const r: Vec2Derived[] = [];
  for (let i = 0; i < ra.pCount; i++) {
    r.push(ra.points[i]);
  }
  return r;
}

export function indexOfLowestPoint(v: Vec2[]): number {
  let i = 0;
  for (let j = 0; j < v.length; j++) {
    if (v[j].y < v[i].y || (v[j].y === v[i].y && v[j].x < v[i].x)) {
      i = j;
    }
  }
  return i;
}

export function minkowskiSumConvex(
  p: ReusablePointArrayLp,
  q: ReusablePointArrayLp,
  // r is expected to contain enough points to hold the result (max p.length+q.length)
  r: ReusablePointArray,
  epsilon = 1e-10,
) {
  let i = p.iLowest;
  let j = q.iLowest;
  let ic = 0;
  let jc = 0;
  let k = 0;

  while (ic < p.pCount || jc < q.pCount) {
    const pa = p.points[i];
    const qa = q.points[j];
    const pb = p.points[i + 1 < p.pCount ? i + 1 : 0];
    const qb = q.points[j + 1 < q.pCount ? j + 1 : 0];
    const pc = {x: pb.x - pa.x, y: pb.y - pa.y};
    const qc = {x: qb.x - qa.x, y: qb.y - qa.y};

    r.points[k].x = pa.x + qa.x;
    r.points[k].y = pa.y + qa.y;
    r.points[k].refA = pa.refA;
    r.points[k].refB = qa.refA;
    k++;

    // "perp dot product"
    let pdp = pc.x * qc.y - pc.y * qc.x;
    if (Math.abs(pdp) < epsilon) pdp = 0;
    if (pdp >= 0) {
      i++;
      ic++;
    }
    if (pdp <= 0) {
      j++;
      jc++;
    }
    if (i === p.pCount) i = 0;
    if (j === q.pCount) j = 0;
  }

  r.pCount = k;
}

export function pointInPolyConvex(
  pt: Vec2,
  poly: ReusablePointArray,
  epsilon = 1e-10,
): boolean {
  // "Always on the negative side" check. Note that negative/positive
  // depends on winding order! "Always on the same side" check
  // would be more general and winding direction would no matter.
  for (let i = 0; i < poly.pCount; i++) {
    const v1 = poly.points[i];
    const v2 = poly.points[i === poly.pCount - 1 ? 0 : i + 1];
    // "perp dot product"
    const pdp = (pt.x - v1.x) * (v2.y - v1.y) - (pt.y - v1.y) * (v2.x - v1.x);
    // 0 = on the line, we count that as inside
    if (pdp > epsilon) return false;
  }

  return true;
}

export function pointPolyConvexMinDist(
  pt: Vec2,
  poly: ReusablePointArray,
  ret: MinDistResult,
  epsilon = 1e-10,
): MinDistResult {
  ret.d = Number.POSITIVE_INFINITY;
  ret.v1i = Number.NaN;
  ret.v2i = Number.NaN;
  let d2: number;
  let mt1: number;
  let mt2: number;
  let mt3: number;
  const t1: Vec2 = {x: 0, y: 0};
  const t2: Vec2 = {x: 0, y: 0};
  const t3: Vec2 = {x: 0, y: 0};
  for (let i = 0; i < poly.pCount; i++) {
    const j = i === poly.pCount - 1 ? 0 : i + 1;
    const v1 = poly.points[i];
    const v2 = poly.points[j];

    t1.x = v2.x - v1.x;
    t1.y = v2.y - v1.y;
    t2.x = pt.x - v1.x;
    t2.y = pt.y - v1.y;

    mt1 = t1.x * t1.x + t1.y * t1.y;
    const r = (t1.x * t2.x + t1.y * t2.y) / mt1;

    if (r < 0) {
      d2 = t2.x * t2.x + t2.y * t2.y;
      if (d2 < ret.d) {
        ret.d = d2;
        ret.v1i = i;
        ret.v2i = Number.NaN;
        ret.minPoint.x = v1.x;
        ret.minPoint.y = v1.y;
      }
    } else if (r > 1) {
      t3.x = pt.x - v2.x;
      t3.y = pt.y - v2.y;
      d2 = t3.x * t3.x + t3.y * t3.y;
      if (d2 < ret.d) {
        ret.d = d2;
        ret.v1i = j;
        ret.v2i = Number.NaN;
        ret.minPoint.x = v2.x;
        ret.minPoint.y = v2.y;
      }
    } else {
      mt2 = t2.x * t2.x + t2.y * t2.y;
      mt3 = r * Math.sqrt(mt1);
      d2 = mt2 - mt3 * mt3;
      // NOTE to self: changing d2 calculation to `d2 = mt2 - r * r * mt1`
      //               and removing mt3 seems to _lower_ performance in FF!
      //               (Maybe about 1-2%).
      // Special handling of d2 = near zero cases
      // (ie. min point ~== pt).
      if (d2 < epsilon) {
        ret.d = 0;
        ret.v1i = i;
        ret.v2i = j;
        ret.minPoint.x = pt.x;
        ret.minPoint.y = pt.y;
      } else if (d2 < ret.d) {
        ret.d = d2;
        ret.v1i = i;
        ret.v2i = j;
        ret.minPoint.x = v1.x + r * t1.x;
        ret.minPoint.y = v1.y + r * t1.y;
      }
    }
  }

  ret.d = Math.sqrt(ret.d);

  return ret;
}

export interface IntersectionPoint extends Vec2 {
  s: number;
  t: number;
  valid: boolean;
}

export function vec2Equal(a: Vec2, b: Vec2, epsilon = 1e-10): boolean {
  return Math.abs(a.x - b.x) < epsilon && Math.abs(a.y - b.y) < epsilon;
}

export function segmentEqual(a: Segment, b: Segment, epsilon = 1e-10): boolean {
  if (vec2Equal(a.a, b.a, epsilon)) {
    return vec2Equal(a.b, b.b, epsilon);
  }
  if (vec2Equal(a.b, b.a, epsilon)) {
    return vec2Equal(a.a, b.b, epsilon);
  }
  return false;
}

export function segmentIntersection(
  a: Segment,
  b: Segment,
  ip: IntersectionPoint,
  tv1: Vec2,
  tv2: Vec2,
  epsilon = 1e-10,
): IntersectionPoint {
  ip.valid = false;
  // Special handling of equal segments: return midpoint of a
  if (segmentEqual(a, b, epsilon)) {
    ip.x = a.a.x + (a.b.x - a.a.x) / 2;
    ip.y = a.a.y + (a.b.y - a.a.y) / 2;
    ip.s = 1;
    ip.t = 0;
    ip.valid = true;
    return ip;
  }
  tv1.x = a.b.x - a.a.x;
  tv1.y = a.b.y - a.a.y;
  tv2.x = b.b.x - b.a.x;
  tv2.y = b.b.y - b.a.y;

  const d = -tv2.x * tv1.y + tv1.x * tv2.y;
  if (d === 0) return ip;

  // TODO: re-think this. if already know that a and b intersect, we only need s because in that case s === t?

  const s = (-tv1.y * (a.a.x - b.a.x) + tv1.x * (a.a.y - b.a.y)) / d;
  if (s < 0 || s > 1) return ip;

  const t = (tv2.x * (a.a.y - b.a.y) - tv2.y * (a.a.x - b.a.x)) / d;
  if (t < 0 || t > 1) return ip;

  ip.x = a.a.x + t * tv1.x;
  ip.y = a.a.y + t * tv1.y;
  ip.s = s;
  ip.t = t;
  ip.valid = true;
  return ip;
}

export function pointToSegmentProjection(
  p: Vec2,
  s: Segment,
  proj: Vec2,
): Vec2 {
  const tv1: Vec2 = {
    x: p.x - s.a.x,
    y: p.y - s.a.y,
  };
  const tv2: Vec2 = {
    x: s.b.x - s.a.x,
    y: s.b.y - s.a.y,
  };

  const dp = tv1.x * tv2.x + tv1.y * tv2.y;
  const mt1 = Math.sqrt(tv1.x * tv1.x + tv1.y * tv1.y);
  const mt2 = Math.sqrt(tv2.x * tv2.x + tv2.y * tv2.y);
  const costh = dp / (mt1 * mt2);
  const r = mt1 * costh;

  proj.x = s.a.x + r * (tv2.x / mt2);
  proj.y = s.a.y + r * (tv2.y / mt2);

  return proj;
}
