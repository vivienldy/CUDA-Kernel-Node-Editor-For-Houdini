vector
vop_curlNoiseVP(vector4 pos, freq, offset; 
		vector nml; 
		string type; string geo;
		int turb, bounce;
		float amp, rough, atten, distance, radius, h)
{
    vector val = {0,0,0};

    if (type == "exact_pnoise")
    {
	return vop_perlinCurlNoiseVP(pos*freq-offset, turb, amp, rough*2, atten);
    }
    else if (type == "exact_xnoise")
    {
	return vop_simplexCurlNoiseVP(pos*freq-offset, turb, amp, rough*2, atten);
    }
    else if (type == "exact_gxnoise")
    {
        return vop_simplexCurlGXNoiseVP(pos*freq-offset, turb, amp, rough*2, atten);
    }

    // Finite difference helpers
    vector4 xDiff = pos;	xDiff.x += h;
    vector4 yDiff = pos;	yDiff.y += h;
    vector4 zDiff = pos;	zDiff.z += h;

    vector noisevec, xDiffNoise, yDiffNoise, zDiffNoise;

    // Noise vectors at and around pos
    if (type == "xnoise")
    {
	noisevec = vop_simplexNoiseVP(pos*freq - offset, turb, amp, rough, atten);
	xDiffNoise = vop_simplexNoiseVP(xDiff*freq - offset, turb, amp, rough, atten);
	yDiffNoise = vop_simplexNoiseVP(yDiff*freq - offset, turb, amp, rough, atten);
	zDiffNoise = vop_simplexNoiseVP(zDiff*freq - offset, turb, amp, rough, atten);
    }
    else
    {
	noisevec = vop_perlinNoiseVP(pos*freq - offset, turb, amp, rough, atten);
	xDiffNoise = vop_perlinNoiseVP(xDiff*freq - offset, turb, amp, rough, atten);
	yDiffNoise = vop_perlinNoiseVP(yDiff*freq - offset, turb, amp, rough, atten);
	zDiffNoise = vop_perlinNoiseVP(zDiff*freq - offset, turb, amp, rough, atten);
    }

    // Ramp function, partial derivatives and cross product
    VOP_CURLNOISE_FUNC()

    return val;
}


vector
vop_simplexNoiseVP(vector4 pos; int turb; float amp, rough, atten)
{
    vector4 pp = pos;
    vector nval;
    float scale = amp;
    int i;
    nval = 0;
    for (int i = 0; i < turb; i++, pp *= 2.0, scale *= rough) nval += 0.5 * scale * ((vector(xnoise(pp))) + -0.5);
    nval = (vector(pow(navl, atten)));
    return nval;
}

VOP_SIMPLEXNOISE_FUNC(vector)