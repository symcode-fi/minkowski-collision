// SPDX-License-Identifier: MIT
// Copyright (c) 2022, Symcode Oy
import rfdc from 'rfdc';
export type {MinDistResult, ReusablePointArray} from './minkowski-math.js';
import {
  minkowskiSumConvex,
  ReusablePointArray,
  pointInPolyConvex,
  Vec2,
  pointPolyConvexMinDist,
  MinDistResult,
  segmentIntersection,
  pointToSegmentProjection,
  Segment,
  IntersectionPoint,
  ReusablePointArrayLp,
  indexOfLowestPoint,
} from './minkowski-math.js';

const cloneDeep = rfdc();

export interface BoundingBox {
  xmin: number;
  xmax: number;
  ymin: number;
  ymax: number;
}

export interface PolyData {
  pos: Vec2;
  v: Vec2[];
}

export interface Body<T extends PolyData> extends PolyData {
  // "Negated" vertices (mirrored through origin)
  vn: Vec2[];
  // Pre-calculated "lowest vertex" index for Minkowski sum calculation
  // NOTE: if rotation should ever be supported these would need to be
  //       updated when body is rotated!
  iLowest: number;
  iLowestN: number;
  nodeId: string;
  index: number;
  bb: BoundingBox;
  data: T;
  castId: number;
}

export interface Collision<T extends PolyData> {
  a: Body<T>;
  b: Body<T>;
}

export interface CastResult<T extends PolyData> {
  // Ray / line collision point
  pos: Vec2;
  b: Body<T>;
}

export interface BodyBodyDistance<T extends PolyData> extends Collision<T> {
  // Distance
  d: number;
  // Distance as vector
  dv: Vec2;
  // Global coordinate of "distance start" point
  pt: Vec2;
}

const ZERO: Vec2 = {
  x: 0,
  y: 0,
};

function polyBoundingBox(poly: PolyData): BoundingBox {
  const bb: BoundingBox = {
    xmin: Number.POSITIVE_INFINITY,
    xmax: Number.NEGATIVE_INFINITY,
    ymin: Number.POSITIVE_INFINITY,
    ymax: Number.NEGATIVE_INFINITY,
  };
  poly.v.forEach((v) => {
    if (v.x < bb.xmin) bb.xmin = v.x;
    if (v.x > bb.xmax) bb.xmax = v.x;
    if (v.y < bb.ymin) bb.ymin = v.y;
    if (v.y > bb.ymax) bb.ymax = v.y;
  });
  return bb;
}

function bbOverlap<T extends PolyData>(a: Body<T>, b: Body<T>): boolean {
  return !(
    a.pos.x + a.bb.xmax < b.pos.x + b.bb.xmin ||
    a.pos.x + a.bb.xmin > b.pos.x + b.bb.xmax ||
    a.pos.y + a.bb.ymax < b.pos.y + b.bb.ymin ||
    a.pos.y + a.bb.ymin > b.pos.y + b.bb.ymax
  );
}
function pointInBb<T extends PolyData>(a: Vec2, b: Body<T>): boolean {
  return !(
    a.x < b.pos.x + b.bb.xmin ||
    a.x > b.pos.x + b.bb.xmax ||
    a.y < b.pos.y + b.bb.ymin ||
    a.y > b.pos.y + b.bb.ymax
  );
}

/**
 * Collision detection with Minkowski difference algorith +
 * simple bounding box optimization.
 */
export default class MinkowskiDiffEngine<T extends PolyData> {
  // Temp body to prefill collisions array
  private someBody: Body<T>;
  public bodies: Body<T>[] = [];
  public collisions: Collision<T>[] = [];
  public collisionCount = 0;
  public castResults: CastResult<T>[] = [];
  public castResultCount = 0;
  private castId = 0;
  private castDistPArr: ReusablePointArray = {points: [], pCount: 0};
  private minBbAxis = 0;
  // Only contains distances from smaller index bodies to larger index bodies.
  // Smaller index (a) defines the row, larger (b) defines the column.
  // Column = b - (a + 1); a === b and a < b are invalid cases.
  public distances: BodyBodyDistance<T>[][] = [];
  // Temp arrays for Minkowski sum input (p and -q)
  private p: ReusablePointArrayLp = {points: [], pCount: 0, iLowest: 0};
  private qn: ReusablePointArrayLp = {points: [], pCount: 0, iLowest: 0};
  // Temp arrays for storing Minkowski sum result.
  public msra: ReusablePointArray[] = [{points: [], pCount: 0}];
  public msrCount = 0;
  private mdr: MinDistResult = {d: 0, v1i: 0, v2i: 0, minPoint: {x: 0, y: 0}};
  // Reusable segmentIntersection() result
  private ip: IntersectionPoint = {
    x: 0,
    y: 0,
    s: 0,
    t: 0,
    valid: false,
  };
  private readonly epsilon = 1e-10;

  // Temporary objects for reuse
  private tv1: Vec2 = {x: 0, y: 0};
  private tv2: Vec2 = {x: 0, y: 0};
  private tv3: Vec2 = {x: 0, y: 0};
  private tv4: Vec2 = {x: 0, y: 0};
  private tv5: Vec2 = {x: 0, y: 0};
  private tv6: Vec2 = {x: 0, y: 0};
  private ts1: Segment = {a: this.tv1, b: this.tv2};
  private ts2: Segment = {a: this.tv1, b: this.tv2};

  constructor(
    private readonly distCalc: boolean,
    private readonly distRefPoint: boolean,
    private readonly storeMsrs: boolean,
    private readonly idGen: (o: T) => string,
    somePolyData: T,
  ) {
    this.someBody = this.createBody(cloneDeep(somePolyData), -1);
  }

  private createBody(somePolyData: T, index: number): Body<T> {
    const vn = somePolyData.v.map((v) => ({x: -v.x, y: -v.y}));
    return {
      pos: somePolyData.pos,
      v: somePolyData.v,
      vn,
      iLowest: indexOfLowestPoint(somePolyData.v),
      iLowestN: indexOfLowestPoint(vn),
      nodeId: this.idGen(somePolyData),
      index,
      bb: polyBoundingBox(somePolyData),
      data: somePolyData,
      castId: -1,
    };
  }

  updateData(data: T[]): void {
    let maxPoints = 0;
    this.bodies.splice(0);
    this.minBbAxis = Number.POSITIVE_INFINITY;

    data.forEach((d, i) => {
      const body = this.createBody(d, i);
      if (body.bb.xmax - body.bb.xmin < this.minBbAxis) {
        this.minBbAxis = body.bb.xmax - body.bb.xmin;
      }
      if (body.bb.ymax - body.bb.ymin < this.minBbAxis) {
        this.minBbAxis = body.bb.ymax - body.bb.ymin;
      }
      this.bodies.push(body);
      maxPoints = Math.max(maxPoints, body.v.length);
      if (i > this.castResults.length) {
        this.castResults.push({
          b: this.someBody,
          pos: {x: 0, y: 0},
        });
      }
    });

    while (this.castDistPArr.points.length < maxPoints) {
      this.castDistPArr.points.push({refA: -1, refB: -1, x: 0, y: 0});
    }

    // Minkowski sum max input point count = Math.max(n, m) = maxPoints
    while (this.p.points.length < maxPoints) {
      this.p.points.push({x: 0, y: 0, refA: Number.NaN, refB: Number.NaN});
      this.qn.points.push({x: 0, y: 0, refA: Number.NaN, refB: Number.NaN});
    }
    // Collisions max count = nth triangular number where:
    //   m = this.bodies.length
    //   n = Math.max(0, m - 1)
    // See:
    //   - https://en.wikipedia.org/wiki/Triangular_number#Formula
    //   - https://en.wikipedia.org/wiki/Triangular_number#Relations_to_other_figurate_numbers
    //     (relation between square and triangular numbers explains the math)
    const n = Math.max(0, this.bodies.length);
    const maxCollisions = (n * (n - 1)) / 2;
    for (let i = 0; i < (this.storeMsrs ? maxCollisions : 1); i++) {
      if (this.msra.length < i + 1) {
        this.msra.push({points: [], pCount: 0});
      }
      const msr = this.msra[i];
      // Minkowski sum max point count = n + m = maxPoints * 2
      while (msr.points.length < maxPoints * 2) {
        msr.points.push({x: 0, y: 0, refA: Number.NaN, refB: Number.NaN});
      }
    }
    while (this.collisions.length < maxCollisions) {
      this.collisions.push({
        a: this.someBody,
        b: this.someBody,
      });
    }
    // Init distances
    for (let i = 0; i < this.bodies.length - 1; i++) {
      const a = this.bodies[i];
      let dists: BodyBodyDistance<T>[] = this.distances[i];
      if (!dists) {
        dists = [];
        this.distances.push(dists);
      }
      for (let j = i + 1; j < this.bodies.length; j++) {
        const b = this.bodies[j];
        const bi = j - (i + 1);
        let dist: BodyBodyDistance<T> = dists[bi];
        if (!dist) {
          dist = {
            a: this.someBody,
            b: this.someBody,
            d: Number.NaN,
            dv: {...ZERO},
            pt: {...ZERO},
          };
          dists.push(dist);
        }
        dist.a = a;
        dist.b = b;
        dist.d = Number.NaN;
        dist.dv.x = 0;
        dist.dv.y = 0;
        dist.pt.y = 0;
        dist.pt.y = 0;
      }
    }
  }

  updateShapePos(id: string, dx: number, dy: number): void {
    const b = this.bodies.find((b) => b.nodeId === id);
    if (b) {
      b.pos.x += dx;
      b.pos.y += dy;
    }
  }

  private updatePQn(ba: Body<T>, bb: Body<T>) {
    // Body a = P
    ba.v.forEach((v, vi) => {
      this.p.points[vi].x = v.x + ba.pos.x;
      this.p.points[vi].y = v.y + ba.pos.y;
    });
    this.p.pCount = ba.v.length;
    this.p.iLowest = ba.iLowest;

    // Body b = -Q (for calculating Minkowski _difference_)
    bb.vn.forEach((v, vi) => {
      this.qn.points[vi].x = v.x - bb.pos.x;
      this.qn.points[vi].y = v.y - bb.pos.y;
    });
    this.qn.pCount = bb.v.length;
    this.qn.iLowest = bb.iLowestN;
  }

  private updatePQnReffed(ba: Body<T>, bb: Body<T>) {
    // Body a = P
    ba.v.forEach((v, vi) => {
      this.p.points[vi].x = v.x + ba.pos.x;
      this.p.points[vi].y = v.y + ba.pos.y;
      this.p.points[vi].refA = vi;
      this.p.points[vi].refB = Number.NaN;
    });
    this.p.pCount = ba.v.length;
    this.p.iLowest = ba.iLowest;

    // Body b = -Q (for calculating Minkowski _difference_)
    bb.vn.forEach((v, vi) => {
      this.qn.points[vi].x = v.x - bb.pos.x;
      this.qn.points[vi].y = v.y - bb.pos.y;
      this.qn.points[vi].refA = vi;
      this.qn.points[vi].refB = Number.NaN;
    });
    this.qn.pCount = bb.v.length;
    this.qn.iLowest = bb.iLowestN;
  }

  pointToBodyMinDist(pt: Vec2, body: Body<T>, ret: MinDistResult): MinDistResult {
    // Put translated points to castDistPArr
    body.v.forEach((v, i) => {
      this.castDistPArr.points[i].x = v.x + body.pos.x;
      this.castDistPArr.points[i].y = v.y + body.pos.y;
    });
    this.castDistPArr.pCount = body.v.length;
    return pointPolyConvexMinDist(pt, this.castDistPArr, ret, this.epsilon);
  }

  lineCast(from: Vec2, to: Vec2) {
    this.castId++;
    this.castResultCount = 0;
    // tv1 = coarse step
    // tv2 = fine step = unit vector
    this.tv1.x = to.x - from.x;
    this.tv1.y = to.y - from.y;
    const d = Math.sqrt(this.tv1.x * this.tv1.x + this.tv1.y * this.tv1.y);
    const step = this.minBbAxis * 0.98;
    let count = d / step + 1 + Number.EPSILON; // +1 because we check the starting point too!
    this.tv2.x = this.tv1.x / d;
    this.tv2.y = this.tv1.y / d;
    this.tv1.x = this.tv2.x * step;
    this.tv1.y = this.tv2.y * step;
    // tv3 = coarse point
    this.tv3.x = from.x;
    this.tv3.y = from.y;

    while (count > 0) {
      // Check polys at tv3
      for (let i = 0; i < this.bodies.length; i++) {
        const b = this.bodies[i];
        if (b.castId == this.castId) continue; // Already hit, ignore

        // If hit, calc this first point hitting nothing, then move
        // from there in small steps towards last coarse point and
        // collect all hits.
        if (pointInBb(this.tv3, b) && this.pointInBody(this.tv3, b)) {
          // tv4 = binary search step vector
          this.tv4.x = this.tv1.x;
          this.tv4.y = this.tv1.y;
          // tv5 = binary search check vector
          this.tv5.x = this.tv3.x;
          this.tv5.y = this.tv3.y;
          let m = -2;

          // Simplified calculus: we go finer until definitely under ½ unit vector.
          let wasHit = true;
          while (Math.abs(this.tv4.x) + Math.abs(this.tv4.y) > 0.5) {
            let hit = false;
            this.tv4.x /= m;
            this.tv4.y /= m;
            m = 2;
            this.tv5.x += this.tv4.x;
            this.tv5.y += this.tv4.y;
            for (let j = 0; j < this.bodies.length; j++) {
              const b2 = this.bodies[j];
              if (b2.castId == this.castId) continue; // Already hit, ignore

              hit =
                hit ||
                (pointInBb(this.tv5, b2) && this.pointInBody(this.tv5, b2));
              // Shortcut: If there was a hit and we're still hitting, gotta keep moving.
              if (wasHit && hit) break;
              // Shortcut: If there wasn't a hit and we're now hitting, gotta turn around
              if (!wasHit && hit) break;
            }
            // If hit status changed, turn around
            if (wasHit != hit) {
              m = -2;
              wasHit = hit;
            }
          }

          // Now go from tv5 towards tv3 using tv2 steps
          const sx = Math.sign(this.tv3.x - this.tv5.x);
          const sy = Math.sign(this.tv3.y - this.tv5.y);
          while (
            Math.sign(this.tv3.x - this.tv5.x) === sx &&
            Math.sign(this.tv3.y - this.tv5.y) === sy
          ) {
            for (let j = 0; j < this.bodies.length; j++) {
              const b2 = this.bodies[j];
              if (b2.castId == this.castId) continue; // Already hit, ignore

              if (pointInBb(this.tv5, b2) && this.pointInBody(this.tv5, b2)) {
                b2.castId = this.castId;
                const r = this.castResults[this.castResultCount++];
                r.b = b2;
                r.pos.x = this.tv5.x;
                r.pos.y = this.tv5.y;
              }
            }
            this.tv5.x += this.tv2.x;
            this.tv5.y += this.tv2.y;
          }
        }
      }

      this.tv3.x += this.tv1.x;
      this.tv3.y += this.tv1.y;
      count--;
    }
  }

  private pointInBody<T extends PolyData>(
    pt: Vec2,
    body: Body<T>,
    epsilon = 1e-10,
  ): boolean {
    // Put translated points to castDistPArr and check
    body.v.forEach((v, i) => {
      this.castDistPArr.points[i].x = v.x + body.pos.x;
      this.castDistPArr.points[i].y = v.y + body.pos.y;
    });
    this.castDistPArr.pCount = body.v.length;
    return pointInPolyConvex(pt, this.castDistPArr, epsilon);
  }

  findBody(nodeId: string): Body<T> | undefined {
    return this.bodies.find((b) => b.nodeId == nodeId);
  }

  checkBodyBodyCollision(ba: Body<T>, bb: Body<T>) {
    this.collisionCount = 0;
    this.msrCount = this.storeMsrs ? 0 : 1;
    if (ba.index < bb.index) {
      this.checkCollision(ba, bb, ba.index, bb.index);
    } else {
      this.checkCollision(bb, ba, bb.index, ba.index);
    }
  }

  checkBodyCollision(ba: Body<T>, stopOnFirst = false) {
    this.collisionCount = 0;
    this.msrCount = this.storeMsrs ? 0 : 1;
    for (let i = 0; i < this.bodies.length; i++) {
      if (i == ba.index) continue;
      const bb = this.bodies[i];
      if (ba.index < bb.index) {
        this.checkCollision(ba, bb, ba.index, bb.index);
      } else {
        this.checkCollision(bb, ba, bb.index, ba.index);
      }
      if (stopOnFirst && this.collisionCount) return;
    }
  }

  checkCollisions(stopOnFirst = false): void {
    this.collisionCount = 0;
    this.msrCount = this.storeMsrs ? 0 : 1;
    for (let i = 0; i < this.bodies.length - 1; i++) {
      const ba = this.bodies[i];
      for (let j = i + 1; j < this.bodies.length; j++) {
        const bb = this.bodies[j];
        this.checkCollision(ba, bb, i, j);
        if (stopOnFirst && this.collisionCount) return;
      }
    }
  }

  private checkCollision(ba: Body<T>, bb: Body<T>, i: number, j: number) {
    // Bounding box check optimization
    if (this.distCalc || bbOverlap(ba, bb)) {
      // NOTE to self: making a sepate `updatePQnFn` where either updatePQn
      //               or updatePQnReffed get assigned to, and using that
      //               in here doesn't seem to help with performance. It may
      //               make it even a little bit worse.
      if (!this.distRefPoint) {
        this.updatePQn(ba, bb);
      } else {
        this.updatePQnReffed(ba, bb);
      }

      const msr = this.msra[this.storeMsrs ? this.msrCount++ : 0];
      minkowskiSumConvex(this.p, this.qn, msr, this.epsilon);
      const collision = pointInPolyConvex(ZERO, msr, this.epsilon);
      if (collision) {
        const coll = this.collisions[this.collisionCount];
        coll.a = ba;
        coll.b = bb;
        this.collisionCount++;
      }

      if (!this.distCalc) return;

      // Calculate and store distance from ba to bb, that's
      // distance from their Minkowski sum to origin.
      const bbd = this.distances[i][j - (i + 1)];
      pointPolyConvexMinDist(ZERO, msr, this.mdr, this.epsilon);
      bbd.d = this.mdr.d;
      // this.mdr.minPoint is vector (origin -> Minkowski sum) which
      // represents distance from bb to ba. Negate to get ba to bb.
      bbd.dv.x = -this.mdr.minPoint.x;
      bbd.dv.y = -this.mdr.minPoint.y;

      if (!this.distRefPoint) return;

      this.updateRefpoints(ba, bb, msr, bbd, collision);
    }
  }

  private updateRefpoints(
    ba: Body<T>,
    bb: Body<T>,
    msr: ReusablePointArray,
    bbd: BodyBodyDistance<T>,
    collision: boolean,
  ) {
    const rpt1 = msr.points[this.mdr.v1i];
    // If this.mdr.v2i is NaN:
    //   (origin -> Minkowski sum) min distance's Minkowski sum
    //   point is on a vertex -> min distance point on ba is
    //   on a vertex. Get Minkowski sum results vertex and
    //   use it's reference to polygon A index to get ba vertex.
    //   Using the same reference point for rpt2 makes us take
    //   the first branch (case a) in if/else statements.
    // else:
    //   (origin -> Minkowski sum) min distance's Minkowski sum
    //   point is on a line segment -> there's 3 possibilities
    //   for min distance between ba and bb:
    //     a) vertex on ba, line segment on bb (same as
    //        this.mdr.v2i being NaN)
    //     b) line segment on ba, vertex on bb
    //     c) line segment on both
    const rpt2 = Number.isNaN(this.mdr.v2i) ? rpt1 : msr.points[this.mdr.v2i];

    if (rpt1.refA === rpt2.refA) {
      // case (a): min distance point on a is on its vertex.
      bbd.pt.x = ba.pos.x + ba.v[rpt1.refA].x;
      bbd.pt.y = ba.pos.y + ba.v[rpt1.refA].y;
    } else if (rpt1.refB === rpt2.refB) {
      // case (b): min distance point on a is on its line segment.
      //           Get point in a by calculating it from point in b,
      //           which is on b's vertex.
      bbd.pt.x = bb.pos.x + bb.v[rpt1.refB].x + this.mdr.minPoint.x;
      bbd.pt.y = bb.pos.y + bb.v[rpt1.refB].y + this.mdr.minPoint.y;
    } else {
      // Case (c): min distance is between line segments of a and b.
      //           That means line segments are parallel. We calculate
      //           min distance point so that min distance vector from
      //           that point goes through intersection point of diagonals
      //           of a rectangle having a's and b's line segments as its
      //           top and bottom. This makes the min distance vector to
      //           "move smoothtly" along both a's and b's line segments
      //           as their alignment changes.
      //           NOTE: "Smooth move" is not really required for collision
      //           detection. It just looks nicer is distance vector is
      //           drawn.
      // In cases where distance is really small a's and b's line segments
      // might not make a rectangle (segments are collinear, or close enough).
      // Solve this by nudging b's segment either more into a or farther from
      // a depending if they've collided or not in the first place.
      // Here t1 is "the nudge".
      this.tv1.x = 0;
      this.tv1.y = 0;
      if (bbd.d < this.epsilon) {
        // A's segment's normal
        this.tv1.x = collision
          ? -(ba.v[rpt2.refA].y - ba.v[rpt1.refA].y)
          : ba.v[rpt2.refA].y - ba.v[rpt1.refA].y;
        this.tv1.y = collision
          ? ba.v[rpt2.refA].x - ba.v[rpt1.refA].x
          : -(ba.v[rpt2.refA].x - ba.v[rpt1.refA].x);
        // Make it small enough to prevent "big visual leaps":
        // divide x and y by longer one.
        const xa = Math.abs(this.tv1.x);
        const ya = Math.abs(this.tv1.y);
        if (xa > ya) {
          this.tv1.x /= xa;
          this.tv1.y /= xa;
        } else {
          this.tv1.x /= ya;
          this.tv1.y /= ya;
        }
      }
      // Get the intersection point of diagonals.
      this.tv2.x = ba.pos.x + ba.v[rpt1.refA].x;
      this.tv2.y = ba.pos.y + ba.v[rpt1.refA].y;
      this.tv3.x = ba.pos.x + ba.v[rpt2.refA].x;
      this.tv3.y = ba.pos.y + ba.v[rpt2.refA].y;
      this.tv4.x = bb.pos.x + bb.v[rpt1.refB].x + this.tv1.x;
      this.tv4.y = bb.pos.y + bb.v[rpt1.refB].y + this.tv1.y;
      this.tv5.x = bb.pos.x + bb.v[rpt2.refB].x + this.tv1.x;
      this.tv5.y = bb.pos.y + bb.v[rpt2.refB].y + this.tv1.y;

      this.ts1.a = this.tv2;
      this.ts1.b = this.tv4;
      this.ts2.a = this.tv3;
      this.ts2.b = this.tv5;

      // We can reuse tv1 here
      segmentIntersection(this.ts1, this.ts2, this.ip, this.tv1, this.tv6);
      if (this.ip.valid) {
        // Project intersection point to A's line segment.
        // NOTE: ts1.a = this.tv2 already!
        // this.ts1.a = this.tv2;
        this.ts1.b = this.tv3;
        // We can reuse tv1 here as result, and tv4 and tv5 as temps
        pointToSegmentProjection(
          this.ip,
          this.ts1,
          this.tv1,
          // this.tv4,
          // this.tv5,
        );
        bbd.pt.x = this.tv1.x;
        bbd.pt.y = this.tv1.y;
      }
    }
  }
}
