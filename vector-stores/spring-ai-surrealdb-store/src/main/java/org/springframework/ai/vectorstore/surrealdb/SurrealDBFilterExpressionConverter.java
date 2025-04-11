/*
 * Copyright 2025 the original honhimW.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.vectorstore.surrealdb;

import org.springframework.ai.vectorstore.filter.Filter;
import org.springframework.ai.vectorstore.filter.converter.AbstractFilterExpressionConverter;

/**
 * @author honhimW
 * @since 1.0.0
 */

public class SurrealDBFilterExpressionConverter extends AbstractFilterExpressionConverter {

	@Override
	protected void doExpression(Filter.Expression expression, StringBuilder context) {
		if (expression.type() == Filter.ExpressionType.NOT) {
			negate(expression, context);
		} else {
			this.convertOperand(expression.left(), context);
			context.append(getOperationSymbol(expression));
			this.convertOperand(expression.right(), context);
		}
	}

	@Override
	protected void doKey(Filter.Key filterKey, StringBuilder context) {
		context.append(filterKey.key());
	}

	private void negate(Filter.Expression expression, StringBuilder context) {
		if (expression.type() == Filter.ExpressionType.NOT) {
			context.append("NOT(");
		}
		this.convertOperand(expression.left(), context);
		context.append(")");
	}

	private String getOperationSymbol(Filter.Expression exp) {
		return switch (exp.type()) {
			case AND -> " AND ";
			case OR -> " OR ";
			case EQ -> " = ";
			case NE -> " != ";
			case LT -> " < ";
			case LTE -> " <= ";
			case GT -> " > ";
			case GTE -> " >= ";
			case IN -> " IN ";
			case NIN -> " NOT IN ";
			// you never know what the future might bring
			default -> throw new RuntimeException("Not supported expression type: " + exp.type());
		};
	}

}
