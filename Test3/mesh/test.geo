Point(1) = {0, 0, 0, 4.0};
Point(2) = {2.5, 0, 0, 4.0};
Point(3) = {2.5, 0.41, 0, 4.0};
Point(4) = {0, 0.41, 0, 4.0};
//+
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
//+

//+ Circle points
r = 0.05;
cx = 0.50;
cy = 0.20;
Point(5) = {cx, cy, 0, 4.0};
Point(6) = {cx+r, cy, 0, 1};
Point(7) = {cx, cy+r, 0, 1};
Point(8) = {cx-r, cy, 0, 1};
Point(9) = {cx, cy-r, 0, 1};
//+
Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};
//+
Curve Loop(1) = {3, 4, 1, 2};
Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(1) = {1, 2};
//+
//Physical Curve("inlet", 1) = {4};
//Physical Curve("outlet", 2) = {2};
//Physical Curve("wall", 3) = {1, 3};
//Physical Curve("cyl", 4) = {5,6,7,8};
//Physical Surface("vol", 5) = {1};
Extrude {0, 0, 0.41} {
  Surface{1}; 
}
// Inlet
Physical Surface(1) = {25}; 
// Outlet
Physical Surface(2) = {33};
// Inner cylinder
Physical Surface(3) = {41, 45, 49, 37};
// Wall y=0
Physical Surface(41) = {29};
// Wall y=Ly
Physical Surface(42) = {21};
// Wall z=0
Physical Surface(43) = {1};
// Wall z=Ly
Physical Surface(44) = {50};
// Volume (required because GMSH)
Physical Volume(5) = {1};
