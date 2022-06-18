/**
 * common.glsl
 * Common types and functions used for ray tracing.
 */

const float pi = 3.14159265358979;
const float epsilon = 0.001;

struct Ray {
    vec3 o;     // origin
    vec3 d;     // direction - always set with normalized vector
    float t;    // time, for motion blur
};

Ray createRay(vec3 o, vec3 d, float t)
{
    Ray r;
    r.o = o;
    r.d = d;
    r.t = t;
    return r;
}

Ray createRay(vec3 o, vec3 d)
{
    return createRay(o, d, 0.0);
}

vec3 pointOnRay(Ray r, float t)
{
    return r.o + r.d * t;
}

float gSeed = 0.0;

uint baseHash(uvec2 p)
{
    p = 1103515245U * ((p >> 1U) ^ (p.yx));
    uint h32 = 1103515245U * ((p.x) ^ (p.y>>3U));
    return h32 ^ (h32 >> 16);
}

float hash1(inout float seed) {
    uint n = baseHash(floatBitsToUint(vec2(seed += 0.1,seed += 0.1)));
    return float(n) / float(0xffffffffU);
}

vec2 hash2(inout float seed) {
    uint n = baseHash(floatBitsToUint(vec2(seed += 0.1,seed += 0.1)));
    uvec2 rz = uvec2(n, n * 48271U);
    return vec2(rz.xy & uvec2(0x7fffffffU)) / float(0x7fffffff);
}

vec3 hash3(inout float seed)
{
    uint n = baseHash(floatBitsToUint(vec2(seed += 0.1, seed += 0.1)));
    uvec3 rz = uvec3(n, n * 16807U, n * 48271U);
    return vec3(rz & uvec3(0x7fffffffU)) / float(0x7fffffff);
}

float rand(vec2 v)
{
    return fract(sin(dot(v.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 toLinear(vec3 c)
{
    return pow(c, vec3(2.2));
}

vec3 toGamma(vec3 c)
{
    return pow(c, vec3(1.0 / 2.2));
}

vec2 randomInUnitDisk(inout float seed) {
    vec2 h = hash2(seed) * vec2(1.0, 6.28318530718);
    float phi = h.y;
    float r = sqrt(h.x);
	return r * vec2(sin(phi), cos(phi));
}

vec3 randomInUnitSphere(inout float seed)
{
    vec3 h = hash3(seed) * vec3(2.0, 6.28318530718, 1.0) - vec3(1.0, 0.0, 0.0);
    float phi = h.y;
    float r = pow(h.z, 1.0/3.0);
	return r * vec3(sqrt(1.0 - h.x * h.x) * vec2(sin(phi), cos(phi)), h.x);
}

vec3 randomUnitVector(inout float seed) //to be used in diffuse reflections with distribution cosine
{
    return(normalize(randomInUnitSphere(seed)));
}

struct Camera
{
    vec3 eye;
    vec3 u, v, n;
    float width, height;
    float lensRadius;
    float planeDist, focusDist;
    float time0, time1;
};

Camera createCamera(
    vec3 eye,
    vec3 at,
    vec3 worldUp,
    float fovy,
    float aspect,
    float aperture,  //diametro em multiplos do pixel size
    float focusDist,  //focal ratio
    float time0,
    float time1)
{
    Camera cam;
    if(aperture == 0.0) cam.focusDist = 1.0; //pinhole camera then focus in on vis plane
    else cam.focusDist = focusDist;
    vec3 w = eye - at;
    cam.planeDist = length(w);
    cam.height = 2.0 * cam.planeDist * tan(fovy * pi / 180.0 * 0.5);
    cam.width = aspect * cam.height;

    cam.lensRadius = aperture * 0.5 * cam.width / iResolution.x;  //aperture ratio * pixel size; (1 pixel=lente raio 0.5)
    cam.eye = eye;
    cam.n = normalize(w); // forward
    cam.u = normalize(cross(worldUp, cam.n)); // right
    cam.v = cross(cam.n, cam.u); // up
    cam.time0 = time0;
    cam.time1 = time1;
    return cam;
}

Ray getRay(Camera cam, vec2 pixel_sample)  //rnd pixel_sample viewport coordinates
{
    float time = cam.time0 + hash1(gSeed) * (cam.time1 - cam.time0);

    // BASE TEST https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/
    // // The ray starts at the camera position (the origin)
    // vec3 rayPosition = cam.eye;
     
    // // calculate coordinates of the ray target on the imaginary pixel plane.
    // // -1 to +1 on x,y axis. 1 unit away on the z axis
    // vec3 rayTarget = vec3((pixel_sample/iResolution.xy) * 2.0 - 1.0, 1.0);
     
    // // calculate a normalized vector for the ray direction.
    // // it's pointing from the ray position to the ray target.
    // vec3 rayDir = normalize(rayTarget - rayPosition);
    // return createRay(rayPosition, rayDir, time);


    // NO DOF
    // vec2 psCoords = (pixel_sample/iResolution.xy) * 2.0 - 1.0;
    // vec2 psCoords = pixel_sample/iResolution.xy;
    // vec3 ray_dir = normalize(cam.width*(psCoords.x-0.5)*cam.u + cam.height*(psCoords.y-0.5)*cam.v - cam.n*cam.planeDist);
    // return createRay(cam.eye, ray_dir, time);

    // DOF
    vec2 ls = cam.lensRadius * randomInUnitDisk(gSeed);  //ls - lens sample for DOF
    
    // ray origin is eye with the lens sample offset
    vec3 eye_offset = cam.eye + cam.u * ls.x + cam.v * ls.y; 

    vec2 psCoords = pixel_sample/iResolution.xy;
    //Point in view plane
    vec3 ps = vec3(cam.width * (psCoords.x - 0.5), cam.height * (psCoords.y - 0.5), 0.0);

    //Point in focal plane
    vec3 p = vec3( ps.x * cam.focusDist, ps.y * cam.focusDist, 0.0);

    // ray direction (p - ls) in world coordinates
    vec3 ray_dir = normalize(vec3( cam.u * (p.x - ls.x) + cam.v * (p.y - ls.y) -cam.n * (cam.planeDist * cam.focusDist) ));

    return createRay(eye_offset, ray_dir, time);
}

// MT_ material type
#define MT_DIFFUSE 0
#define MT_METAL 1
#define MT_DIALECTRIC 2

struct Material
{
    int type;
    vec3 albedo;  //diffuse color
    vec3 specColor;  //the color tint for specular reflections. for metals and opaque dieletrics like coloured glossy plastic
    vec3 emissive; //
    float roughness; // controls roughness for metals. It can be used for rough refractions
    float refIdx; // index of refraction for dialectric
    vec3 refractColor; // absorption for beer's law
};

Material createDiffuseMaterial(vec3 albedo)
{
    Material m;
    m.type = MT_DIFFUSE;
    m.albedo = albedo;
    m.specColor = vec3(0.0);
    m.roughness = 1.0;  //ser usado na iluminação direta
    m.refIdx = 1.0;
    m.refractColor = vec3(0.0);
    m.emissive = vec3(0.0);
    return m;
}

Material createDiffuseEmissiveMaterial(vec3 albedo, vec3 emissive)
{
    Material m;
    m.type = MT_DIFFUSE;
    m.albedo = albedo;
    m.specColor = vec3(0.0);
    m.roughness = 1.0;  
    m.refIdx = 1.0;
    m.refractColor = vec3(0.0);
    m.emissive = emissive;
    return m;
}

Material createMetalMaterial(vec3 specClr, float roughness)
{
    Material m;
    m.type = MT_METAL;
    m.albedo = vec3(0.0);
    m.specColor = specClr;
    m.roughness = roughness;
    m.emissive = vec3(0.0);
    return m;
}

Material createDialectricMaterial(vec3 refractClr, float refIdx, float roughness)
{
    Material m;
    m.type = MT_DIALECTRIC;
    m.albedo = vec3(0.0);
    m.specColor = vec3(0.04);
    m.refIdx = refIdx;
    m.refractColor = refractClr;  
    m.roughness = roughness;
    m.emissive = vec3(0.0);
    return m;
}

struct HitRecord
{
    vec3 pos;
    vec3 normal;
    float t;            // ray parameter
    Material material;
};


float schlick(float cosine, float refIdx)
{
    float r0 = (1.-refIdx)/(1.+refIdx);
    r0 *= r0;
    return r0 + (1.-r0)*pow((1.-cosine),5.);
}

vec3 metalSchlick(float cosine, vec3 F0)
{
    return F0 + (1.0-F0) * pow(clamp(1.0 - cosine, 0.0, 1.0), 5.0);
}

vec3 beer(vec3 color, float distanceTravelled)
{
    vec3 absorbance = (vec3(1.)-color) * 1.0 * -distanceTravelled;
    return exp(absorbance); 
}

bool customRefract(vec3 rIn_direction, vec3 outwardNormal, float niOverNt, out vec3 refracted)
{
    float cosI;
    vec3 v;
    vec3 vt;
    float sinT;
    float sinI;
    float cosT;
    float aux;
    vec3 t;
    vec3 rt;

    cosI = dot(rIn_direction, outwardNormal);

    v = - rIn_direction;

    vt = (dot(v,normalize(outwardNormal))) * normalize(outwardNormal) - v;

    sinI = length(vt);

    sinT = niOverNt*sinI;

    t = (1.0 / length(vt)) * vt;

    aux = 1.0 - (sinT * sinT);
    cosT = sqrt(aux);

    rt = sinT * t + cosT * ( -normalize(outwardNormal));

    refracted = rt;

    if (sinT > 1.0){
        return false;
    }

    return true;
}


bool scatter(Ray rIn, HitRecord rec, out vec3 atten, out Ray rScattered)
{
    if(rec.material.type == MT_DIFFUSE) // Lambertian scattering
    {
        //INSERT CODE HERE,
        vec3 rsTarget = rec.pos + rec.normal + randomInUnitSphere(gSeed);
        vec3 rd = normalize(rsTarget - rec.pos);
        vec3 ro = rec.pos + rec.normal * epsilon;
        rScattered = createRay(ro, rd, rIn.t);
        atten = rec.material.albedo * max(dot(rScattered.d, rec.normal), 0.0) / pi;
        return true;
    }
    if(rec.material.type == MT_METAL)
    {
       //INSERT CODE HERE, consider fuzzy reflections
        vec3 rd = reflect(rIn.d, rec.normal);
        vec3 ro = rec.pos + rec.normal * epsilon;
        rScattered = createRay(ro, normalize(rd + rec.material.roughness*randomInUnitSphere(gSeed)), rIn.t);
        atten = metalSchlick(-dot(rIn.d, rec.normal), rec.material.specColor);
        return true;
    }
    if(rec.material.type == MT_DIALECTRIC)
    {
        atten = vec3(1.0);
        vec3 outwardNormal;
        float niOverNt;
        float cosine;

        cosine = -dot(rIn.d, rec.normal);

        if(cosine < 0.0) //hit inside
        {
            outwardNormal = -rec.normal;
            niOverNt = rec.material.refIdx;
            atten = beer(rec.material.refractColor, rec.t); // Beer's law
        }
        else  //hit from outside
        {
            outwardNormal = rec.normal;
            niOverNt = 1.0 / rec.material.refIdx;
        }

        //Use probabilistic math to decide if scatter a reflected ray or a refracted ray

        vec3 refracted;
        float reflectProb;

        if(customRefract(rIn.d, outwardNormal, niOverNt, refracted)){ // no total internal reflection
            if (niOverNt > 1.0){ // inside material, leaving
                cosine = sqrt(1.0 - niOverNt*niOverNt*(1.0 - cosine*cosine));
            }
            reflectProb = schlick(cosine, rec.material.refIdx);
        }
        else
            reflectProb = 1.;

        if ( hash1(gSeed) < reflectProb )  //Reflection
        {
            vec3 reflected = reflect(rIn.d, outwardNormal);
            rScattered = createRay(rec.pos + epsilon * rec.normal, normalize(reflected), rIn.t);

        } 
        else //Refraction
        {
            rScattered = createRay(rec.pos - epsilon * outwardNormal, normalize(refracted), rIn.t);
        }

        return true;
    }
    return false;
}

struct Triangle {vec3 a; vec3 b; vec3 c; };

Triangle createTriangle(vec3 v0, vec3 v1, vec3 v2)
{
    Triangle t;
    t.a = v0; t.b = v1; t.c = v2;
    return t;
}

bool hit_triangle(Triangle tr, Ray r, float tmin, float tmax, out HitRecord rec)
{
    vec3 v1 = tr.b - tr.a;
    vec3 v2 = tr.c - tr.a;
    vec3 v3 = -r.d;

    vec3 VS = r.o - tr.a;

    // Cramer's rule
    float d = 1.0/determinant(mat3(v1, v2, v3 ));
    float u =   d*determinant(mat3(VS, v2, v3 ));
    float v =   d*determinant(mat3(v1, VS, v3 ));
    float t =   d*determinant(mat3(v1, v2, VS));

    if( u<0.0 || v<0.0 || (u+v)>1.0 )
        return false;

	if (t > tmin && t < tmax)
    {
        rec.t = t;
        rec.normal = normalize(cross(tr.b - tr.a, tr.c - tr.a));
        rec.pos = pointOnRay(r, rec.t);
        return true;
    }
    return false;
}

struct Sphere
{
    vec3 center;
    float radius;
};

Sphere createSphere(vec3 center, float radius)
{
    Sphere s;
    s.center = center;
    s.radius = radius;
    return s;
}


struct MovingSphere
{
    vec3 center0, center1;
    float radius;
    float time0, time1;
};

MovingSphere createMovingSphere(vec3 center0, vec3 center1, float radius, float time0, float time1)
{
    MovingSphere s;
    s.center0 = center0;
    s.center1 = center1;
    s.radius = radius;
    s.time0 = time0;
    s.time1 = time1;
    return s;
}

vec3 center(MovingSphere mvsphere, float time)
{
    return mvsphere.center0 + (mvsphere.center1 - mvsphere.center0) * ((time - mvsphere.time0) / (mvsphere.time1 - mvsphere.time0));
}


/*
 * The function naming convention changes with these functions to show that they implement a sort of interface for
 * the book's notion of "hittable". E.g. hit_<type>.
 */

bool hit_sphere(Sphere s, Ray r, float tmin, float tmax, out HitRecord rec)
{
    //get the vector from the center of this sphere to where the ray begins.
	vec3 m = r.o - s.center;

    //get the dot product of the above vector and the ray's vector
	float b = dot(m, r.d);

	float c = dot(m, m) - s.radius * s.radius;

	//exit if r's origin outside s (c > 0) and r pointing away from s (b > 0)
	if(c > 0.0 && b > 0.0)
		return false;

	//calculate discriminant
	float discr = b * b - c;

	//a negative discriminant corresponds to ray missing sphere
	if(discr < 0.0)
		return false;
    
	//ray now found to intersect sphere, compute smallest t value of intersection
    bool fromInside = false;
	float dist = -b - sqrt(discr);
    if (dist < 0.0)
    {
        fromInside = true;
        dist = -b + sqrt(discr);
    }
    
	if (dist > tmin && dist < tmax)
    {
        rec.t = dist;
        rec.pos = pointOnRay(r, rec.t);
        rec.normal = normalize(rec.pos - s.center); 

        return true;
    }
    
    return false;
}

// bool hit_movingSphere(MovingSphere s, Ray r, float tmin, float tmax, out HitRecord rec)
// {
//     float B, C, delta;
//     bool outside;
//     float t;


//      //INSERT YOUR CODE HERE
//      //Calculate the moving center
//     //calculate a valid t and normal
	
//     if(t < tmax && t > tmin) {
//         rec.t = t;
//         rec.pos = pointOnRay(r, rec.t);
//         rec.normal = normal;
//         return true;
//     }
//     else return false;
// }

struct pointLight {
    vec3 pos;
    vec3 color;
};

pointLight createPointLight(vec3 pos, vec3 color) 
{
    pointLight l;
    l.pos = pos;
    l.color = color;
    return l;
}