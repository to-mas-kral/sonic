#ifndef PT_SPECTRAL_DATA_H
#define PT_SPECTRAL_DATA_H

#include "../utils/basic_types.h"
#include "sampled_spectrum.h"

// TODO: figure out static GPU data

/// Data from: https://refractiveindex.info/?shelf=glass&book=BK7&page=SCHOTT (public
/// domain)
/*const UmVector<f32> GLASS_BK7_ETA_RAW = UmVector{
    300.f,
    322.f,
    344.f,
    366.f,
    388.f,
    410.f,
    432.f,
    454.f,
    476.f,
    498.f,
    520.f,
    542.f,
    564.f,
    586.f,
    608.f,
    630.f,
    652.f,
    674.f,
    696.f,
    718.f,
    740.f,
    762.f,
    784.f,
    806.f,
    828.f,
    850.f,
    872.f,
    894.f,
    916.f,

    1.5527702635739f,
    1.5458699289209f,
    1.5404466868331f,
    1.536090527917f,
    1.53252773217f,
    1.529568767224f,
    1.5270784291406f,
    1.5249578457324f,
    1.5231331738499f,
    1.5215482528369f,
    1.5201596882463f,
    1.5189334783109f,
    1.5178426478869f,
    1.516865556749f,
    1.5159846691816f,
    1.5151856452759f,
    1.5144566604975f,
    1.513787889767f,
    1.5131711117948f,
    1.5125994024544f,
    1.5120668948646f,
    1.5115685899969f,
    1.5111002059336f,
    1.5106580569705f,
    1.5102389559626f,
    1.5098401349174f,
    1.5094591800239f,
    1.5090939781792f,
    1.5087426727363f,
};*/

const ConstantSpectrum AIR_ETA = ConstantSpectrum::make(1.000277f);

const ConstantSpectrum GLASS_BK7_ETA = ConstantSpectrum::make(1.530277f);

const ConstantSpectrum POLYPROPYLENE_ETA = ConstantSpectrum::make(1.49f);

/*const PiecewiseSpectrum GLASS_BK7_ETA = PiecewiseSpectrum::make(
    CSpan<f32>(GLASS_BK7_ETA_RAW.get_ptr(), GLASS_BK7_ETA_RAW.size()));*/

#endif // PT_SPECTRAL_DATA_H
