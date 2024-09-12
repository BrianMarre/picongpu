/* Copyright 2024 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/** @file implements conversion from TemperatureFunctorTag to TemperatureFunctor
 *
 * @attention Each TemperatureFunctor must have exactly one associated tag and one associated conversion implementation
 */

#pragma once

#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stewartPyatt/temperatureFunctor/ClassicalTemperatureFunctor.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stewartPyatt/temperatureFunctor/RelativisticTemperatureFunctor.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stewartPyatt/temperatureFunctor/TemperatureFunctorTags.hpp"

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::stewartPyatt::temperatureFunctor
{
    //! default case, must create specialization for every TemperatureFunctor
    template<typename TemperatureFunctorTag>
    struct TagToFunctor;

    template<>
    struct TagToFunctor<ClassicalTemperatureFunctorTag>
    {
        using type = ClassicalTemperatureFunctor;
    }

    template<>
    struct TagToFunctor<RelativisticTemperatureFunctorTag>
    {
        using type = RelativisticTemperatureFunctor;
    }
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::stewartPyatt::temperatureFunctor
