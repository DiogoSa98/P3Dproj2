/**
* ver hash functions em
* https://www.shadertoy.com/view/XlGcRh hash functions GPU
* http://www.jcgt.org/published/0009/03/02/
 */

 #include "./common.glsl"
 #iChannel0 "self"

bool hit_world(Ray r, float tmin, float tmax, out HitRecord rec)
{
    bool hit = false;
    rec.t = tmax;
   
    if(hit_triangle(createTriangle(vec3(-10.0, -0.01, 10.0), vec3(10.0, -0.01, 10.0), vec3(-10.0, -0.01, -10.0)), r, tmin, rec.t, rec))
    {
        hit = true;
        rec.material = createDiffuseMaterial(vec3(0.2));
    }

    if(hit_triangle(createTriangle(vec3(-10.0, -0.01, -10.0), vec3(10.0, -0.01, 10), vec3(10.0, -0.01, -10.0)), r, tmin, rec.t, rec))
    {
        hit = true;
        rec.material = createDiffuseMaterial(vec3(0.2));
    }

    if(hit_sphere(
        createSphere(vec3(-4.0, 1.0, 0.0), 1.0),
        r,
        tmin,
        rec.t,
        rec))
    {
        hit = true;
        rec.material = createDiffuseMaterial(vec3(0.2, 0.95, 0.1));
        //rec.material = createDiffuseMaterial(vec3(0.4, 0.2, 0.1));
    }

    if(hit_sphere(
        createSphere(vec3(4.0, 1.0, 0.0), 1.0),
        r,
        tmin,
        rec.t,
        rec))
    {
        hit = true;
        rec.material = createMetalMaterial(vec3(0.7, 0.6, 0.5), 0.0);
    }

    if(hit_sphere(
        createSphere(vec3(0.0, 1.0, 0.0), 1.0),
        r,
        tmin,
        rec.t,
        rec))
    {
        hit = true;
        rec.material = createDialectricMaterial(vec3(0.0), 1.333, 0.0);
    }

// if(hit_sphere(
//         createSphere(vec3(0.0, 1.0, 0.0), -0.95),
//         r,
//         tmin,
//         rec.t,
//         rec))
//     {
//         hit = true;
//         rec.material = createDialectricMaterial(vec3(0.0), 1.333, 0.0);
//     }
   
    int numxy = 5;
    
    for(int x = -numxy; x < numxy; ++x)
    {
        for(int y = -numxy; y < numxy; ++y)
        {
            float fx = float(x);
            float fy = float(y);
            float seed = fx + fy / 1000.0;
            vec3 rand1 = hash3(seed);
            vec3 center = vec3(fx + 0.9 * rand1.x, 0.2, fy + 0.9 * rand1.y);
            float chooseMaterial = rand1.z;
            if(distance(center, vec3(4.0, 0.2, 0.0)) > 0.9)
            {
                if(chooseMaterial < 0.3)
                {
                    vec3 center1 = center + vec3(0.0, hash1(gSeed) * 0.5, 0.0);
                    // diffuse
                    // if(hit_movingSphere(
                    //     createMovingSphere(center, center1, 0.2, 0.0, 1.0),
                    //     r,
                    //     tmin,
                    //     rec.t,
                    //     rec))
                    // {
                    //     hit = true;
                    //     rec.material = createDiffuseMaterial(hash3(seed) * hash3(seed));
                    // }
                }
                else if(chooseMaterial < 0.5)
                {
                    // diffuse
                    if(hit_sphere(
                        createSphere(center, 0.2),
                        r,
                        tmin,
                        rec.t,
                        rec))
                    {
                        hit = true;
                        rec.material = createDiffuseMaterial(hash3(seed) * hash3(seed));
                    }
                }
                else if(chooseMaterial < 0.7)
                {
                    // metal
                    if(hit_sphere(
                        createSphere(center, 0.2),
                        r,
                        tmin,
                        rec.t,
                        rec))
                    {
                        hit = true;
                       // rec.material.type = MT_METAL;
                        rec.material = createMetalMaterial((hash3(seed) + 1.0) * 0.5, 0.0);
                    }
                }
                else if(chooseMaterial < 0.9)
                {
                    // metal
                    if(hit_sphere(
                        createSphere(center, 0.2),
                        r,
                        tmin,
                        rec.t,
                        rec))
                    {
                        hit = true;
                       // rec.material.type = MT_METAL;
                        rec.material = createMetalMaterial((hash3(seed) + 1.0) * 0.5, hash1(seed));
                    }
                }
                else
                {
                    // glass (dialectric)
                    if(hit_sphere(
                        createSphere(center, 0.2),
                        r,
                        tmin,
                        rec.t,
                        rec))
                    {
                        hit = true;
                        rec.material.type = MT_DIALECTRIC;
                        rec.material = createDialectricMaterial(hash3(seed), 1.2, 0.0);
                    }
                }
            }
        }
    }
    return hit;
}

vec3 directlighting(pointLight pl, Ray r, HitRecord rec){
    vec3 diffCol, specCol;
    vec3 colorOut = vec3(0.0, 0.0, 0.0);
    float shininess;
    HitRecord dummy;

    //INSERT YOUR CODE HERE
    vec3 lDir = normalize(pl.pos - rec.pos);
    float intensity = max(dot(rec.normal, lDir), 0.0);
    Ray shadowRay = createRay(rec.pos + rec.normal * epsilon, lDir);
    float lDist = length(pl.pos - rec.pos);

    if (intensity > 0.0) {
        if (!hit_world(shadowRay, 0.0, lDist, dummy)) {
            if (rec.material.type == MT_DIFFUSE) {
                diffCol = rec.material.albedo / pi * intensity;
                specCol = rec.material.specColor;
                shininess = 4.0 / (pow(rec.material.roughness, 4.0)+epsilon)-2.0; // no idea what is going on here
            }
            if (rec.material.type == MT_METAL) {
                diffCol = rec.material.albedo;
                specCol = metalSchlick(intensity, rec.material.specColor);
                shininess = 8.0 / (pow(rec.material.roughness, 4.0)+epsilon)-2.0; // no idea what is going on here
            }
            if (rec.material.type == MT_DIALECTRIC) {
                diffCol = rec.material.albedo;
                specCol = rec.material.specColor;
                shininess = 500.0; 
            }

            vec3 h = normalize(lDir - r.d);
            float intSpec = max(dot(h, rec.normal), 0.0);
            vec3 spec = specCol * pow(intSpec, shininess);
            colorOut = (diffCol + spec) * pl.color;
        }
    }

	return colorOut; 
}

#define MAX_BOUNCES 10

vec3 rayColor(Ray r)
{
    HitRecord rec;
    vec3 col = vec3(0.0);
    vec3 throughput = vec3(1.0, 1.0, 1.0);
    for(int i = 0; i < MAX_BOUNCES; ++i)
    {
        if(hit_world(r, 0.001, 10000.0, rec))
        {
            //calculate direct lighting with 3 white point lights:
            {
                col += directlighting(createPointLight(vec3(-10.0, 15.0, 0.0), vec3(1.0, 1.0, 1.0)), r, rec) * throughput;
                col += directlighting(createPointLight(vec3(8.0, 15.0, 3.0), vec3(1.0, 1.0, 1.0)), r, rec) * throughput;
                col += directlighting(createPointLight(vec3(1.0, 15.0, -9.0), vec3(1.0, 1.0, 1.0)), r, rec) * throughput;
            }
            
            //calculate secondary ray and update throughput
            Ray scatterRay;
            vec3 atten;
            if(scatter(r, rec, atten, scatterRay))
            {   
                //  insert your code here
                r = scatterRay;
                throughput *= atten;  
            } 
            else
            {
                return vec3(0.0);
            }
        
        }
        else  //background
        {
            float t = 0.8 * (r.d.y + 1.0);
            col += throughput * mix(vec3(1.0), vec3(0.5, 0.7, 1.0), t);
            break;
        }
    }
    return col;
}

#define MAX_SAMPLES 10000.0

void main()
{
    gSeed = float(baseHash(floatBitsToUint(gl_FragCoord.xy))) / float(0xffffffffU) + iTime;

    vec2 mouse = iMouse.xy / iResolution.xy;
    mouse.x = mouse.x * 2.0 - 1.0;

    vec3 camPos = vec3(mouse.x * 10.0, mouse.y * 5.0, 8.0);
    vec3 camTarget = vec3(0.0, 0.0, -1.0);
    float fovy = 60.0;
    float aperture = 7.0;
    float distToFocus = 1.0;
    float time0 = 0.0;
    float time1 = 1.0;
    Camera cam = createCamera(
        camPos,
        camTarget,
        vec3(0.0, 1.0, 0.0),    // world up vector
        fovy,
        iResolution.x / iResolution.y,
        aperture,
        distToFocus,
        time0,
        time1);

//usa-se o 4 canal de cor para guardar o numero de samples e não o iFrame pois quando se mexe o rato faz-se reset

    vec4 prev = texture(iChannel0, gl_FragCoord.xy / iResolution.xy);
    vec3 prevLinear = toLinear(prev.xyz);  

    vec2 ps = gl_FragCoord.xy + hash2(gSeed);
    //vec2 ps = gl_FragCoord.xy;
    vec3 color = rayColor(getRay(cam, ps));

    if(iMouseButton.x != 0.0 || iMouseButton.y != 0.0)
    {
        gl_FragColor = vec4(toGamma(color), 1.0);  //samples number reset = 1
        return;
    }
    if(prev.w > MAX_SAMPLES)   
    {
        gl_FragColor = prev;
        return;
    }

    float w = prev.w + 1.0;
    color = mix(prevLinear, color, 1.0/w);
    gl_FragColor = vec4(toGamma(color), w);
}
