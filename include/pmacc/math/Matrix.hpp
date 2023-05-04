/* Copyright 2023 Brian Marre
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

//#include <picongpu/param/atomicPhysics2_pmacc_Debug.param>


#include "pmacc/math/Vector.hpp"
#include "pmacc/memory/Array.hpp"

#include <cstdint>

namespace pmacc::math
{
    /** common stuff general interface
     *
     * @tparam T_Type data type used for storage of elements
     * @tparam T_order number of dimensions of matrix(so called order),
     *      for example MxN matrix has order 2
     * @tparam T_extent pmacc::math::CT::Vector, extent of matrix in each dimensions
     */
    template<typename T_Type, typename T_Extent, bool PMACC_MATRIX_HOT_DEBUG = true>
    struct Matrix
    {
        using ThisType = Matrix<T_Type, T_Extent>;
        using Type = T_Type;
        using Extent = T_Extent;
        using S_Elements = pmacc::memory::Array<T_Type, CT::volume<T_Extent>::type::value>;
        using Idx = typename S_Elements::size_type;

    private:
        S_Elements elements;

        /** get linear memory storage index from n-dimensional index
         *
         * indexation with the following scheme for a matrix with extent = < <N1>, <N2> >
         *
         * #(linear Index) | 0 | 1 | ... | <N1>-1 | <N1> | <N1> + 1 | ... | <N1>+<N1>-1 | 2 * <N1>
         * ----------------|---|---|-----|--------|------|----------|-----|-------------|---------
         *                m| 0 | 1 | ... | <N1>-1 |    0 |        1 | ... |  (<N1> - 1) | 0, ...
         *                n| 0 | 0 | ... |      0 |    1 |        1 | ... |           1 | 2, ...
         *
         * @attention no range checks outside debug, invalid input will lead to illegal memory access!
         */
        HDINLINE static Idx getLinearIndex(uint32_t const& m, uint32_t const& n)
        {
            // debug range check
            if constexpr(PMACC_MATRIX_HOT_DEBUG)
            {
                if(m >= T_Extent::template at<0u>::type::value)
                {
                    printf("PMACC_ERROR: invalid index m for matrix access!\n");
                    return static_cast<uint32_t>(0u);
                }
                if(n >= T_Extent::template at<1u>::type::value)
                {
                    printf("PMACC_ERROR: invalid index n for matrix access!\n");
                    return static_cast<uint32_t>(0u);
                }
            }

            return m + n * T_Extent::template at<1u>::type::value;
        }

    public:
        //! constructor, @attention leaves elements uninitialized!
        Matrix()
        {
            PMACC_CASSERT_MSG(not_a_matrix, T_Extent::dim == 2u);
        }

        Matrix(pmacc::math::Vector<T_Type, T_Extent::template at<0u>::type::value> const& vector)
        {
            PMACC_CASSERT(T_Extent::template at<1u>::type::value == 1u);

#pragma unroll
            for(uint32_t i = static_cast<uint32_t>(0u); i < T_Extent::template at<0u>::type::value; i++)
                this->element(i, static_cast<uint32_t>(0u)) = vector[i];
        }

        /** access via (m,n)
         *
         * @attention idx indexation starts with 0!
         * @attention no range checks outside debug compile!, invalid idx will result in
         *  illegal memory access!
         */
        HDINLINE T_Type& element(uint32_t const m, uint32_t const n)
        {
            return elements[getLinearIndex(m, n)];
        }

        /** access via (m,n), const version
         *
         * @attention idx indexation starts with 0!
         * @attention no range checks outside debug compile!, invalid idx will result in
         *  illegal memory access!
         */
        HDINLINE T_Type const& element(uint32_t const m, uint32_t const n) const
        {
            return elements[getLinearIndex(m, n)];
        }

        //! access via linear index, @attention no range check outside debug compile!
        HDINLINE T_Type& element(Idx const idx)
        {
            if constexpr(PMACC_MATRIX_HOT_DEBUG)
            {
                if(idx >= CT::volume<T_Extent>::type::value)
                {
                    printf("PMACC_ERROR: invalid linear index in Matrix operator[] call\n");
                    return elements[static_cast<uint32_t>(0u)];
                }
            }
            return elements[idx];
        }

        //! access via linear index, const version
        HDINLINE T_Type const& element(Idx const idx) const
        {
            return elements[idx];
        }

        //! matrix multiplication, A x B = C
        template<typename T_ExtentRhs>
        HDINLINE void mMul(
            Matrix<T_Type, T_ExtentRhs> const& rhs,
            Matrix<
                T_Type,
                pmacc::math::CT::
                    Vector<typename T_Extent::template at<0u>::type, typename T_ExtentRhs::template at<1u>::type>>&
                result) const
        {
            using ExtentA = T_Extent;
            using ExtentB = T_ExtentRhs;
            using ExtentC = pmacc::math::CT::
                Vector<typename T_Extent::template at<0u>::type, typename T_ExtentRhs::template at<1u>::type>;

            PMACC_CASSERT(ExtentB::dim == 2u);
            PMACC_CASSERT(ExtentA::template at<1u>::type::value == ExtentB::template at<0u>::type::value);

#pragma unroll
            for(uint32_t i = static_cast<uint32_t>(0u); i < ExtentA::template at<0u>::type::value; i++)
            {
#pragma unroll
                for(uint32_t j = static_cast<uint32_t>(0u); j < ExtentB::template at<1u>::type::value; j++)
                {
                    T_Type& c_ij = result.element(i, j);

                    // reset before we add up
                    c_ij = static_cast<T_Type>(0.);

#pragma unroll
                    for(uint32_t n = static_cast<uint32_t>(0u); n < ExtentA::template at<1u>::type::value; n++)
                    {
                        c_ij += this->element(i, n) * rhs.element(n, j);
                    }
                }
            }
        }

        //! component wise multiplication
        HDINLINE ThisType& sMul(T_Type const a)
        {
#pragma unroll
            for(uint32_t i = static_cast<uint32_t>(0u); i < T_Extent::template at<0u>::type::value; i++)
            {
#pragma unroll
                for(uint32_t j = static_cast<uint32_t>(0u); j < T_Extent::template at<1u>::type::value; j++)
                {
                    this->element(i, j) = this->element(i, j) * a;
                }
            }
            return *this;
        }

        template<typename T_ExtentRhs>
        HDINLINE ThisType& operator+(Matrix<T_Type, T_ExtentRhs> const& rhs)
        {
            using ExtentA = T_Extent;
            using ExtentB = T_ExtentRhs;

            PMACC_CASSERT(ExtentA::template at<0u>::type::value == ExtentB::template at<0u>::type::value);
            PMACC_CASSERT(ExtentA::template at<1u>::type::value == ExtentB::template at<1u>::type::value);

#pragma unroll
            for(uint32_t i = static_cast<uint32_t>(0u); i < ExtentA::template at<0u>::type::value; i++)
            {
#pragma unroll
                for(uint32_t j = static_cast<uint32_t>(0u); j < ExtentB::template at<1u>::type::value; j++)
                {
                    this->element(i, j) = this->element(i, j) + rhs.element(i, j);
                }
            }

            return *this;
        }
    };
} // namespace pmacc::math
